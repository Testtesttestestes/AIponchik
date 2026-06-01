"""Trainable move-ranking policy for the Cookie Cats style board.

Powered by PyTorch for blazing fast CUDA training and batched inference.
"""

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from game_parser import MoveCandidate, MovePathfinder, RandomGameSimulator, SimulationConfig, ensure_parent_dir

ITEMS = RandomGameSimulator.ITEMS
TARGETS = (*ITEMS, "ice")
FEATURE_NAMES = (
    "move_length_norm",
    "move_ice_fraction",
    "move_item_target_fraction",
    "move_ice_target_fraction",
    "moves_left_norm",
    "path_bbox_height_norm",
    "path_bbox_width_norm",
    "path_centroid_row_norm",
    "path_centroid_col_norm",
    "future_cluster_score_norm",
    "future_orphan_fraction",
    "future_ice_target_norm",
    "future_cluster_count_norm",
    "own_target_urgency",
    "ice_target_urgency",
    "max_other_target_urgency",
    *(f"move_item_{item}" for item in ITEMS),
    *(f"target_remaining_{target}" for target in TARGETS),
    *(f"board_fraction_{item}" for item in ITEMS),
    "board_ice_fraction",
)


@dataclass(frozen=True)
class TrainingConfig:
    train_games: int = 60
    val_games: int = 20
    candidates_per_state: int = 16
    epochs: int = 80
    hidden_units: int = 128
    learning_rate: float = 0.003
    l2: float = 0.0005
    dropout: float = 0.1
    patience: int = 15
    seed: int = 20260601
    ice_hits: int = 3
    batch_size: int = 512


@dataclass
class PolicyDataset:
    x: np.ndarray
    y: np.ndarray
    state_ids: np.ndarray
    best_candidate_rows: np.ndarray

    @property
    def size(self):
        return int(self.x.shape[0])

    @property
    def state_count(self):
        return int(len(np.unique(self.state_ids))) if self.state_ids.size else 0


class PyTorchMLP(nn.Module):
    """The underlying PyTorch architecture."""
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
    """Encode a board state and one candidate move into numeric features."""
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

        row_values = [row for row, _ in path]
        col_values = [col for _, col in path]
        row_norm = max(1, rows - 1)
        col_norm = max(1, cols - 1)
        next_board = self.future_evaluator.simulate_after_move(board, path)
        future = self.future_evaluator.evaluate(next_board, state.get("gameState", {}).get("targets", {}))
        
        urgencies = {
            name: (data.get("remaining", 0) / moves_left) if moves_left else 0.0
            for name, data in targets.items()
        }
        other_urgencies = [value for name, value in urgencies.items() if name not in {move.item, "ice"}]

        values: List[float] = [
            path_len / cells,
            path_ice / path_len,
            own_collected / max(1, own_remaining),
            ice_collected / max(1, ice_remaining),
            moves_left / 40.0,
            (max(row_values) - min(row_values) + 1) / max(1, rows),
            (max(col_values) - min(col_values) + 1) / max(1, cols),
            (sum(row_values) / path_len) / row_norm,
            (sum(col_values) / path_len) / col_norm,
            future.cluster_score / 12.0,
            future.orphan_count / cells,
            future.ice_target_score / 10.0,
            future.cluster_count / cells,
            min(1.5, urgencies.get(move.item, 0.0)) / 1.5,
            min(1.5, urgencies.get("ice", 0.0)) / 1.5,
            min(1.5, max(other_urgencies) if other_urgencies else 0.0) / 1.5,
        ]
        values.extend(1.0 if move.item == item else 0.0 for item in ITEMS)
        values.extend(targets.get(target, {}).get("remaining", 0) / (10.0 if target == "ice" else 35.0) for target in TARGETS)
        values.extend(board_counts[item] / cells for item in ITEMS)
        values.append(board_ice / cells)
        return np.asarray(values, dtype=np.float32)


class NeuralMovePolicy:
    """Wrapper around PyTorch model to handle normalization and JSON I/O."""

    def __init__(self, input_size: int, hidden_units: int = 128, seed: int = 0, dropout: float = 0.1):
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PyTorchMLP(input_size, hidden_units, dropout).to(self.device)
        self.mean = torch.zeros(input_size, device=self.device)
        self.std = torch.ones(input_size, device=self.device)

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            if x_tensor.ndim == 1:
                x_tensor = x_tensor.unsqueeze(0)
            x_standardized = (x_tensor - self.mean) / self.std
            pred = self.model(x_standardized)
        return pred.cpu().numpy()

    def fit(self, train: PolicyDataset, val: PolicyDataset, config: TrainingConfig, save_path: str = None):
        print(f"[{'GPU' if self.device.type == 'cuda' else 'CPU'}] Starting PyTorch training...")
        
        # Move data to GPU
        x_train = torch.tensor(train.x, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(train.y, dtype=torch.float32, device=self.device)
        x_val = torch.tensor(val.x, dtype=torch.float32, device=self.device)
        y_val = torch.tensor(val.y, dtype=torch.float32, device=self.device)

        # Standardize directly on GPU
        self.mean = x_train.mean(dim=0)
        self.std = torch.clamp(x_train.std(dim=0), min=1e-4)

        x_train = (x_train - self.mean) / self.std
        x_val = (x_val - self.mean) / self.std

        # DataLoader for fast batched processing
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.l2)
        criterion = nn.MSELoss()

        best_val = float("inf")
        stale_epochs = 0
        history = []
        best_state = None

        for epoch in range(1, config.epochs + 1):
            self.model.train()
            train_loss_accum = 0.0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item() * batch_x.size(0)

            train_loss = train_loss_accum / len(train_dataset)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(x_val)
                val_loss = criterion(val_pred, y_val).item()
                val_pred_np = val_pred.cpu().numpy()
                val_top1 = top1_agreement(val_pred_np, val.y, val.state_ids)
            
            history.append({"epoch": epoch, "trainLoss": train_loss, "valLoss": val_loss, "valTop1Agreement": val_top1})

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                stale_epochs = 0
                
                if save_path:
                    self.model.load_state_dict(best_state)
                    self.save(save_path)
            else:
                stale_epochs += 1
                if stale_epochs >= config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            
        return {
            "epochsRun": len(history),
            "bestValLoss": best_val,
            "history": history,
            "trainTop1Agreement": top1_agreement(self.predict(train.x), train.y, train.state_ids),
            "valTop1Agreement": top1_agreement(self.predict(val.x), val.y, val.state_ids),
        }

    def to_dict(self, metrics=None, config=None):
        # Convert PyTorch tensors to basic python lists for JSON serialization
        state_dict = self.model.state_dict()
        network_json = {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}
        
        return {
            "format": "aiponchik-pytorch-policy-v2",
            "featureNames": list(FEATURE_NAMES),
            "standardization": {"mean": self.mean.cpu().numpy().tolist(), "std": self.std.cpu().numpy().tolist()},
            "network": network_json,
            "trainingConfig": config.__dict__ if config else None,
            "metrics": metrics or {},
        }

    @classmethod
    def from_dict(cls, payload: Dict):
        if payload.get("featureNames") != list(FEATURE_NAMES):
            raise ValueError("Policy feature list does not match this code version")
            
        network = payload["network"]
        # Derive sizes from weights
        input_size = len(payload["featureNames"])
        hidden_units = len(network["net.0.bias"])
        
        policy = cls(input_size, hidden_units)
        policy.mean = torch.tensor(payload["standardization"]["mean"], device=policy.device)
        policy.std = torch.tensor(payload["standardization"]["std"], device=policy.device)
        
        # Load weights into PyTorch model
        state_dict = {k: torch.tensor(v, device=policy.device) for k, v in network.items()}
        policy.model.load_state_dict(state_dict)
        return policy

    def save(self, path: str, metrics=None, config=None):
        ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(metrics=metrics, config=config), handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


class PolicyTrainingDatasetBuilder:
    """Create train/validation samples from simulated games with expert labels."""
    def __init__(self, etalon_dir="etalon_images", seed=20260601, ice_hits=3, candidates_per_state=16):
        config = SimulationConfig(seed=seed, ice_hits=ice_hits)
        self.simulator = RandomGameSimulator(etalon_dir=etalon_dir, config=config, pathfinder=MovePathfinder(max_paths_per_item=20, rollout_samples=0))
        self.encoder = MoveFeatureEncoder()
        self.candidates_per_state = candidates_per_state

    def build(self, games: int, seed_offset: int = 0) -> PolicyDataset:
        self.simulator.rng = np.random.default_rng(self.simulator.config.seed + seed_offset)
        features = []
        labels = []
        state_ids = []
        best_rows = []
        state_id = 0
        
        print(f"Building dataset for {games} games...")
        for game_index in range(1, games + 1):
            for state, candidates in self._iter_teacher_states(game_index):
                if len(candidates) < 2:
                    continue
                selected = candidates[: self.candidates_per_state]
                scores = np.asarray([move.score for move in selected], dtype=np.float32)
                score_span = float(scores.max() - scores.min())
                normalized = np.ones_like(scores) if score_span < 1e-6 else (scores - scores.min()) / score_span
                start_row = len(features)
                for move, label in zip(selected, normalized):
                    features.append(self.encoder.encode(state, move))
                    labels.append(float(label))
                    state_ids.append(state_id)
                best_rows.append(start_row + int(np.argmax(normalized)))
                state_id += 1
                
        if not features:
            raise RuntimeError("Could not build a policy dataset: no candidate moves found")
            
        print(f"Dataset compiled: {len(features)} move candidates across {state_id} board states.")
        return PolicyDataset(
            x=np.vstack(features).astype(np.float32),
            y=np.asarray(labels, dtype=np.float32),
            state_ids=np.asarray(state_ids, dtype=np.int32),
            best_candidate_rows=np.asarray(best_rows, dtype=np.int32),
        )

    def _iter_teacher_states(self, game_index: int):
        template = self.simulator.etalon_states[int(self.simulator.rng.integers(0, len(self.simulator.etalon_states)))]
        difficulty = list(self.simulator.DIFFICULTIES)[(game_index - 1) % len(self.simulator.DIFFICULTIES)]
        modifiers = self.simulator.DIFFICULTIES[difficulty]
        level = str(template.get("gameState", {}).get("level", "0"))
        moves_limit = self.simulator._moves_limit(template, modifiers)
        targets_remaining = self.simulator._scaled_targets(template, modifiers)
        board, ice_hp = self.simulator._random_board_from_template(template, modifiers)
        for turn in range(1, moves_limit + 1):
            if self.simulator._targets_done(targets_remaining):
                break
            state = self.simulator._state_for_solver(board, ice_hp, targets_remaining, moves_limit - turn + 1, level)
            paths = self.simulator.pathfinder.find_paths(state["board"])
            candidates = self.simulator.pathfinder.score_paths(
                state["board"],
                state["gameState"]["targets"],
                paths,
                moves_left=moves_limit - turn + 1,
            )
            if not candidates:
                board, ice_hp = self.simulator._force_reseed_playable_area(board, ice_hp)
                continue
            yield state, candidates
            best_move = candidates[0]
            self.simulator._apply_move(board, ice_hp, best_move.path, targets_remaining)
            self.simulator._refill_board(board, ice_hp)


def top1_agreement(predictions: np.ndarray, labels: np.ndarray, state_ids: np.ndarray) -> float:
    wins = 0
    states = 0
    for state_id in np.unique(state_ids):
        rows = np.flatnonzero(state_ids == state_id)
        if rows.size == 0:
            continue
        states += 1
        if rows[int(np.argmax(predictions[rows]))] == rows[int(np.argmax(labels[rows]))]:
            wins += 1
    return round(wins / states * 100.0, 2) if states else 0.0


def train_policy(config: TrainingConfig, etalon_dir="etalon_images", save_path=None, resume_path=None):
    builder = PolicyTrainingDatasetBuilder(
        etalon_dir=etalon_dir,
        seed=config.seed,
        ice_hits=config.ice_hits,
        candidates_per_state=config.candidates_per_state,
    )
    train = builder.build(config.train_games, seed_offset=0)
    val = builder.build(config.val_games, seed_offset=100_000)
    
    # --- НОВАЯ ЛОГИКА ЗАГРУЗКИ ---
    if resume_path:
        print(f"🔄 Загрузка существующих весов из {resume_path}...")
        model = NeuralMovePolicy.load(resume_path)
        # Убедимся, что загруженная модель отправлена на правильное устройство (GPU)
        model.model = model.model.to(model.device)
    else:
        print("✨ Создание новой модели с нуля...")
        model = NeuralMovePolicy(len(FEATURE_NAMES), hidden_units=config.hidden_units, seed=config.seed, dropout=config.dropout)
    # -----------------------------
    
    metrics = model.fit(train, val, config, save_path=save_path)
    
    metrics.update({
        "trainSamples": train.size,
        "valSamples": val.size,
        "trainStates": train.state_count,
        "valStates": val.state_count,
    })
    return model, metrics


def select_policy_move(model: NeuralMovePolicy, encoder: MoveFeatureEncoder, state: Dict, candidates: Sequence[MoveCandidate], blend_heuristic: float = 0.0):
    if not candidates:
        return None
    features = np.vstack([encoder.encode(state, move) for move in candidates]).astype(np.float32)
    neural_scores = model.predict(features)
    
    if blend_heuristic > 0.0:
        heuristic_scores = np.asarray([move.score for move in candidates], dtype=np.float32)
        span = float(heuristic_scores.max() - heuristic_scores.min())
        if span > 1e-6:
            heuristic_scores = (heuristic_scores - heuristic_scores.min()) / span
        else:
            heuristic_scores = np.zeros_like(heuristic_scores)
            
        neural_span = float(neural_scores.max() - neural_scores.min())
        if neural_span > 1e-6:
            neural_scores = (neural_scores - neural_scores.min()) / neural_span
            
        combined = (1.0 - blend_heuristic) * neural_scores + blend_heuristic * heuristic_scores
    else:
        combined = neural_scores
        
    return candidates[int(np.argmax(combined))]


def evaluate_policy_games(
    model: NeuralMovePolicy,
    games: int = 50,
    seed: int = 20260602,
    etalon_dir: str = "etalon_images",
    candidates_per_state: int = 16,
    blend_heuristic: float = 0.0,
):
    simulator = RandomGameSimulator(
        etalon_dir=etalon_dir,
        config=SimulationConfig(seed=seed),
        pathfinder=MovePathfinder(max_paths_per_item=20, rollout_samples=0),
    )
    encoder = MoveFeatureEncoder()
    results = []
    
    for game_index in range(1, games + 1):
        template = simulator.etalon_states[int(simulator.rng.integers(0, len(simulator.etalon_states)))]
        difficulty = list(simulator.DIFFICULTIES)[(game_index - 1) % len(simulator.DIFFICULTIES)]
        modifiers = simulator.DIFFICULTIES[difficulty]
        level = str(template.get("gameState", {}).get("level", "0"))
        moves_limit = simulator._moves_limit(template, modifiers)
        targets_remaining = simulator._scaled_targets(template, modifiers)
        board, ice_hp = simulator._random_board_from_template(template, modifiers)
        turns = 0
        
        for turn in range(1, moves_limit + 1):
            if simulator._targets_done(targets_remaining):
                break
            state = simulator._state_for_solver(board, ice_hp, targets_remaining, moves_limit - turn + 1, level)
            paths = simulator.pathfinder.find_paths(state["board"])
            candidates = simulator.pathfinder.score_paths(
                state["board"],
                state["gameState"]["targets"],
                paths,
                moves_left=moves_limit - turn + 1,
            )[:candidates_per_state]
            
            if not candidates:
                board, ice_hp = simulator._force_reseed_playable_area(board, ice_hp)
                continue
                
            move = select_policy_move(model, encoder, state, candidates, blend_heuristic=blend_heuristic)
            simulator._apply_move(board, ice_hp, move.path, targets_remaining)
            simulator._refill_board(board, ice_hp)
            turns += 1
            
        results.append({
            "gameIndex": game_index,
            "difficulty": difficulty,
            "success": simulator._targets_done(targets_remaining),
            "movesUsed": turns,
            "movesLimit": moves_limit,
            "targetsRemaining": dict(targets_remaining),
        })
        
    wins = sum(1 for result in results if result["success"])
    by_difficulty = {}
    for result in results:
        bucket = by_difficulty.setdefault(result["difficulty"], {"games": 0, "wins": 0})
        bucket["games"] += 1
        bucket["wins"] += int(result["success"])
        
    for bucket in by_difficulty.values():
        bucket["successRate"] = round(bucket["wins"] / bucket["games"] * 100.0, 2) if bucket["games"] else 0.0
        
    return {
        "games": games,
        "wins": wins,
        "losses": games - wins,
        "successRate": round(wins / games * 100.0, 2) if games else 0.0,
        "byDifficulty": by_difficulty,
        "details": results,
    }
