"""Trainable move-ranking policy for the Cookie Cats style board.
Phase 2: Discrimination Contrast (Pairwise Loss) & Calibrated Metrics.
"""

import json
import os
import concurrent.futures
from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from game_parser import MoveCandidate, MovePathfinder, RandomGameSimulator, SimulationConfig, ensure_parent_dir

ITEMS = RandomGameSimulator.ITEMS
TARGETS = (*ITEMS, "ice")
FEATURE_NAMES = (
    "move_length_norm", "move_ice_fraction", "move_item_target_fraction", "move_ice_target_fraction",
    "moves_left_norm", "path_bbox_height_norm", "path_bbox_width_norm", "path_centroid_row_norm",
    "path_centroid_col_norm", "future_cluster_score_norm", "future_orphan_fraction",
    "future_ice_target_norm", "future_cluster_count_norm", "own_target_urgency",
    "ice_target_urgency", "max_other_target_urgency",
    *(f"move_item_{item}" for item in ITEMS),
    *(f"target_remaining_{target}" for target in TARGETS),
    *(f"board_fraction_{item}" for item in ITEMS),
    "board_ice_fraction",
)

@dataclass(frozen=True)
class TrainingConfig:
    train_games: int = 500
    val_games: int = 100
    candidates_per_state: int = 32
    epochs: int = 1
    hidden_units: int = 128
    learning_rate: float = 1e-4  
    l2: float = 0.0005
    dropout: float = 0.1
    patience: int = 10
    seed: int = 20260601
    ice_hits: int = 3
    batch_size: int = 512
    epsilon: float = 0.15
    gamma: float = 0.99  
    # PHASE 2: Pairwise Learning params
    pairwise_weight: float = 0.3 # Вес вспомогательного лосса
    margin: float = 0.1 # Желаемый зазор между хорошим и плохим действием
    temperature: float = 1.0 # Для расчета энтропии Q-значений

@dataclass
class PolicyDataset:
    x: np.ndarray
    y: np.ndarray
    x_alt: np.ndarray # Альтернативные (невыбранные) действия
    weights: np.ndarray # Recency weights
    chosen_indices: np.ndarray = None 
    q_entropy_mean: float = 0.0
    q_margin_mean: float = 0.0

    @property
    def size(self):
        return int(self.x.shape[0])

@dataclass
class GameSampleBuffer:
    features: List[np.ndarray]
    labels: List[float]
    features_alt: List[np.ndarray]
    chosen_indices: List[int]
    q_entropies: List[float]
    q_margins: List[float]
    success: bool

class PyTorchMLP(nn.Module):
    def __init__(self, input_size: int, hidden_units: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, max(16, hidden_units // 2)),
            nn.GELU(),
            nn.Linear(max(16, hidden_units // 2), 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class MoveFeatureEncoder:
    def __init__(self):
        self.pathfinder = MovePathfinder(max_paths_per_item=20, rollout_samples=0)
        self.future_evaluator = self.pathfinder.future_evaluator

    def encode(self, state: Dict, move: MoveCandidate) -> np.ndarray:
        board = state.get("board", [])
        rows = len(board)
        cols = len(board[0]) if rows else 0
        cells = max(1, rows * cols)
        path = tuple(move.path)
        path_len = max(1, len(path))
        targets = self.pathfinder._normalize_targets(state.get("gameState", {}).get("targets", {}))
        moves_left = self.pathfinder._parse_moves_left(state.get("gameState", {}).get("movesLeft")) or 0

        board_counts = {item: 0 for item in ITEMS}
        board_ice = 0
        for row in board:
            for cell in row:
                item = MovePathfinder.base_item(cell)
                if item in board_counts:
                    board_counts[item] += 1
                if MovePathfinder.has_ice(cell):
                    board_ice += 1

        path_ice = sum(1 for row, col in path if MovePathfinder.has_ice(board[row][col]))
        own_remaining = targets.get(move.item, {}).get("remaining", 0)
        ice_remaining = targets.get("ice", {}).get("remaining", 0)
        own_collected = min(path_len, own_remaining) if own_remaining else 0
        ice_collected = min(path_ice, ice_remaining) if ice_remaining else 0

        row_values = [r for r, _ in path]
        col_values = [c for _, c in path]
        next_board = self.future_evaluator.simulate_after_move(board, path)
        future = self.future_evaluator.evaluate(next_board, targets)
        
        urgencies = {name: (data.get("remaining", 0) / moves_left) if moves_left else 0.0 for name, data in targets.items()}
        other_urgencies = [val for name, val in urgencies.items() if name not in {move.item, "ice"}]

        values: List[float] = [
            path_len / cells, path_ice / path_len, own_collected / max(1, own_remaining),
            ice_collected / max(1, ice_remaining), moves_left / 40.0,
            (max(row_values) - min(row_values) + 1) / max(1, rows),
            (max(col_values) - min(col_values) + 1) / max(1, cols),
            (sum(row_values) / path_len) / max(1, rows - 1),
            (sum(col_values) / path_len) / max(1, cols - 1),
            future.cluster_score / 12.0, future.orphan_count / cells, future.ice_target_score / 10.0,
            future.cluster_count / cells, min(1.5, urgencies.get(move.item, 0.0)) / 1.5,
            min(1.5, urgencies.get("ice", 0.0)) / 1.5, min(1.5, max(other_urgencies) if other_urgencies else 0.0) / 1.5,
        ]
        values.extend(1.0 if move.item == item else 0.0 for item in ITEMS)
        values.extend(targets.get(tgt, {}).get("remaining", 0) / (10.0 if tgt == "ice" else 35.0) for tgt in TARGETS)
        values.extend(board_counts[item] / cells for item in ITEMS)
        values.append(board_ice / cells)
        return np.asarray(values, dtype=np.float32)

class NeuralMovePolicy:
    def __init__(self, input_size: int, hidden_units: int = 128, seed: int = 0, dropout: float = 0.1):
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PyTorchMLP(input_size, hidden_units, dropout).to(self.device)
        self.mean = torch.zeros(input_size, device=self.device)
        self.std = torch.ones(input_size, device=self.device)
        self.is_normalized = False

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            if x_tensor.ndim == 1: x_tensor = x_tensor.unsqueeze(0)
            x_standardized = (x_tensor - self.mean) / self.std
            pred = self.model(x_standardized)
        return pred.cpu().numpy()

    def fit(self, train: PolicyDataset, config: TrainingConfig):
        if train.size == 0:
            return {"epochsRun": 0, "trainLoss": float("inf"), "mean_grad_norm": 0.0, "max_grad_norm": 0.0}

        x_train = torch.tensor(train.x, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(train.y, dtype=torch.float32, device=self.device)
        x_alt = torch.tensor(train.x_alt, dtype=torch.float32, device=self.device)
        weights = torch.tensor(train.weights, dtype=torch.float32, device=self.device)
        
        if not self.is_normalized and torch.all(self.mean == 0.0):
            self.mean = x_train.mean(dim=0)
            self.std = torch.clamp(x_train.std(dim=0), min=1e-4)
            self.is_normalized = True

        x_train = (x_train - self.mean) / self.std
        x_alt = (x_alt - self.mean) / self.std
        
        train_loader = DataLoader(TensorDataset(x_train, y_train, x_alt, weights), batch_size=config.batch_size, shuffle=True)
        optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.l2)
        
        # reduction='none' нужен для применения custom weights и auxiliary loss
        criterion = nn.SmoothL1Loss(beta=0.2, reduction='none')

        self.model.train()
        train_loss_accum = 0.0
        val_loss_accum = 0.0
        pair_loss_accum = 0.0
        total_grad_norm = 0.0
        max_grad_norm = 0.0
        steps = 0
        
        for epoch in range(config.epochs):
            for b_x, b_y, b_alt, b_w in train_loader:
                optimizer.zero_grad()
                
                # Main Objective: Value Regression
                q_chosen = self.model(b_x)
                loss_value = criterion(q_chosen, b_y)
                loss_value = (loss_value * b_w).mean()
                
                # Auxiliary Objective: Pairwise Ranking
                q_alt = self.model(b_alt)
                is_diff = (~torch.all(b_x == b_alt, dim=1)).float() # Игнорируем, если ход единственный
                loss_pairwise = torch.relu(config.margin - (q_chosen - q_alt)) * is_diff
                loss_pairwise = (loss_pairwise * b_w).mean()

                loss = loss_value + config.pairwise_weight * loss_pairwise
                loss.backward()
                
                batch_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None: batch_norm += p.grad.detach().data.norm(2).item() ** 2
                batch_norm = batch_norm ** 0.5
                total_grad_norm += batch_norm
                max_grad_norm = max(max_grad_norm, batch_norm)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss_accum += loss.item() * b_x.size(0)
                val_loss_accum += loss_value.item() * b_x.size(0)
                pair_loss_accum += loss_pairwise.item() * b_x.size(0)
                steps += 1

        return {
            "epochsRun": config.epochs, 
            "trainLoss": train_loss_accum / train.size,
            "valueLoss": val_loss_accum / train.size,
            "pairLoss": pair_loss_accum / train.size,
            "mean_grad_norm": total_grad_norm / max(1, steps),
            "max_grad_norm": max_grad_norm
        }

    def to_dict(self, metrics=None, config=None):
        return {
            "format": "aiponchik-pytorch-policy-v5-rl-pairwise",
            "featureNames": list(FEATURE_NAMES),
            "standardization": {"mean": self.mean.cpu().numpy().tolist(), "std": self.std.cpu().numpy().tolist(), "is_normalized": self.is_normalized},
            "network": {k: v.cpu().numpy().tolist() for k, v in self.model.state_dict().items()},
            "trainingConfig": config.__dict__ if config else None,
            "metrics": metrics or {},
        }

    @classmethod
    def from_dict(cls, payload: Dict):
        policy = cls(len(payload["network"]["net.0.weight"][0]), len(payload["network"]["net.0.bias"]))
        policy.mean = torch.tensor(payload["standardization"]["mean"], device=policy.device)
        policy.std = torch.tensor(payload["standardization"]["std"], device=policy.device)
        policy.is_normalized = payload["standardization"].get("is_normalized", True)
        policy.model.load_state_dict({k: torch.tensor(v, device=policy.device) for k, v in payload["network"].items()})
        return policy

    def save(self, path: str, metrics=None, config=None):
        ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as h: json.dump(self.to_dict(metrics=metrics, config=config), h, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as h: return cls.from_dict(json.load(h))

class SelfPlayDatasetBuilder:
    def __init__(self, etalon_dir="etalon_images", config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.simulator = RandomGameSimulator(etalon_dir=etalon_dir, config=SimulationConfig(seed=self.config.seed, ice_hits=self.config.ice_hits), pathfinder=MovePathfinder(max_paths_per_item=20, rollout_samples=0))
        self.encoder = MoveFeatureEncoder()
        self.model_payload = None

    def set_model(self, model: Optional[NeuralMovePolicy]):
        self.model_payload = model.to_dict() if model else None

    def _build_single_game(self, game_index: int, seed_offset: int):
        from game_parser import SimulationConfig, RandomGameSimulator, MovePathfinder
        
        local_simulator = RandomGameSimulator(self.simulator.etalon_dir, config=SimulationConfig(seed=self.simulator.config.seed + seed_offset + game_index, ice_hits=self.simulator.config.ice_hits), pathfinder=MovePathfinder(max_paths_per_item=20, rollout_samples=0))
        local_encoder = MoveFeatureEncoder()

        local_model = None
        if self.model_payload:
            local_model = NeuralMovePolicy.from_dict(self.model_payload)
            local_model.device = torch.device("cpu")
            local_model.model.to("cpu")
            local_model.mean = local_model.mean.to("cpu")
            local_model.std = local_model.std.to("cpu")
            local_model.model.eval()

        g_feats, g_labels, g_feats_alt, g_chosen, g_entropies, g_margins = [], [], [], [], [], []
        template = local_simulator.etalon_states[int(local_simulator.rng.integers(0, len(local_simulator.etalon_states)))]
        modifiers = local_simulator.DIFFICULTIES[list(local_simulator.DIFFICULTIES)[(game_index - 1) % len(local_simulator.DIFFICULTIES)]]
        moves_limit = local_simulator._moves_limit(template, modifiers)
        targets_remaining = local_simulator._scaled_targets(template, modifiers)
        board, ice_hp = local_simulator._random_board_from_template(template, modifiers)

        for turn in range(1, moves_limit + 1):
            if local_simulator._targets_done(targets_remaining): break
            state = local_simulator._state_for_solver(board, ice_hp, targets_remaining, moves_limit - turn + 1, str(template.get("gameState", {}).get("level", "0")))
            candidates = local_simulator.pathfinder.score_paths(state["board"], state["gameState"]["targets"], local_simulator.pathfinder.find_paths(state["board"]), moves_left=moves_limit - turn + 1)
            
            if not candidates:
                board, ice_hp = local_simulator._force_reseed_playable_area(board, ice_hp)
                continue

            selected = candidates[: self.config.candidates_per_state]
            is_random = local_simulator.rng.random() < self.config.epsilon
            q_ent, q_marg = 0.0, 0.0
            
            if local_model is not None:
                features_all = np.vstack([local_encoder.encode(state, m) for m in selected]).astype(np.float32)
                q_values = local_model.predict(features_all)
                
                # ИНСТРУМЕНТИРОВАНИЕ: Расчет Q-Энтропии и Margin
                if len(q_values) > 1:
                    q_sorted = np.sort(q_values)[::-1]
                    q_marg = q_sorted[0] - q_sorted[1]
                    exp_q = np.exp((q_values - np.max(q_values)) / self.config.temperature)
                    probs = exp_q / np.sum(exp_q)
                    q_ent = -np.sum(probs * np.log(probs + 1e-8))
                
                chosen_idx = int(local_simulator.rng.integers(0, len(selected))) if is_random else int(np.argmax(q_values))
            else:
                chosen_idx = int(local_simulator.rng.integers(0, len(selected)))

            chosen_move = selected[chosen_idx]
            # Выбираем случайную альтернативу для Pairwise Loss
            alt_choices = [m for i, m in enumerate(selected) if i != chosen_idx]
            alt_move = alt_choices[int(local_simulator.rng.integers(0, len(alt_choices)))] if alt_choices else chosen_move

            g_feats.append(local_encoder.encode(state, chosen_move))
            g_feats_alt.append(local_encoder.encode(state, alt_move))
            g_labels.append(0.0) 
            g_chosen.append(chosen_idx)
            g_entropies.append(q_ent)
            g_margins.append(q_marg)

            local_simulator._apply_move(board, ice_hp, chosen_move.path, targets_remaining)
            local_simulator._refill_board(board, ice_hp)

        success = local_simulator._targets_done(targets_remaining)
        
        final_reward = 1.0 if success else -1.0
        steps = len(g_labels)
        for i in range(steps):
            distance_to_end = steps - 1 - i
            g_labels[i] = final_reward * (self.config.gamma ** distance_to_end)

        return GameSampleBuffer(g_feats, g_labels, g_feats_alt, g_chosen, g_entropies, g_margins, success)

    def build(self, games: int, seed_offset: int = 0) -> PolicyDataset:
        import multiprocessing as mp
        cores = max(1, min(os.cpu_count() or 1, games))
        a_f, a_l, a_falt, a_idx, a_ent, a_marg = [], [], [], [], [], []
        successful_games = 0

        ctx = mp.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=cores, mp_context=ctx) as executor:
            futures = [executor.submit(self._build_single_game, i, seed_offset) for i in range(1, games + 1)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    game = future.result()
                    successful_games += int(game.success)
                    if game.features:
                        a_f.extend(game.features)
                        a_l.extend(game.labels)
                        a_falt.extend(game.features_alt)
                        a_idx.extend(game.chosen_indices)
                        a_ent.extend(game.q_entropies)
                        a_marg.extend(game.q_margins)
                except Exception as e:
                    print(f"Worker Error in Self-Play: {e}")

        ds = PolicyDataset(
            x=np.vstack(a_f).astype(np.float32) if a_f else np.empty((0, len(FEATURE_NAMES))),
            y=np.asarray(a_l, dtype=np.float32) if a_l else np.empty((0,)),
            x_alt=np.vstack(a_falt).astype(np.float32) if a_falt else np.empty((0, len(FEATURE_NAMES))),
            weights=np.ones(len(a_l), dtype=np.float32), # Defaults to 1.0
            chosen_indices=np.asarray(a_idx, dtype=np.int32) if a_idx else np.empty((0,)),
            q_entropy_mean=float(np.mean(a_ent)) if a_ent else 0.0,
            q_margin_mean=float(np.mean(a_marg)) if a_marg else 0.0
        )
        ds.successful_games = successful_games
        return ds

def evaluate_policy_games(model: NeuralMovePolicy, games: int, etalon_dir: str, candidates: int = 32):
    simulator = RandomGameSimulator(etalon_dir, config=SimulationConfig(seed=777), pathfinder=MovePathfinder(max_paths_per_item=20, rollout_samples=0))
    encoder = MoveFeatureEncoder()
    wins = 0
    for i in range(games):
        template = simulator.etalon_states[int(simulator.rng.integers(0, len(simulator.etalon_states)))]
        mod = simulator.DIFFICULTIES[list(simulator.DIFFICULTIES)[i % len(simulator.DIFFICULTIES)]]
        board, ice = simulator._random_board_from_template(template, mod)
        rem = simulator._scaled_targets(template, mod)
        lim = simulator._moves_limit(template, mod)
        for t in range(1, lim + 1):
            if simulator._targets_done(rem): break
            st = simulator._state_for_solver(board, ice, rem, lim - t + 1, "0")
            cands = simulator.pathfinder.score_paths(board, rem, simulator.pathfinder.find_paths(board), moves_left=lim - t + 1)[:candidates]
            if not cands: 
                board, ice = simulator._force_reseed_playable_area(board, ice)
                continue
            feats = np.vstack([encoder.encode(st, c) for c in cands]).astype(np.float32)
            simulator._apply_move(board, ice, cands[int(np.argmax(model.predict(feats)))].path, rem)
            simulator._refill_board(board, ice)
        wins += int(simulator._targets_done(rem))
    return round(wins / games * 100.0, 2) if games else 0.0