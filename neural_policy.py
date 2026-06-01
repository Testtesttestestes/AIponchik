"""Trainable move-ranking policy for the Cookie Cats style board.

This module is intentionally separate from ``game_parser.py``.  The analyzer
continues to use the hand-written heuristic until a trained policy is promoted
explicitly; here we only build, train, save, and evaluate the neural policy.
"""

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

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
    """Hyperparameters chosen to reduce overfitting on small simulated data."""

    train_games: int = 60
    val_games: int = 20
    candidates_per_state: int = 16
    epochs: int = 80
    hidden_units: int = 48
    learning_rate: float = 0.003
    l2: float = 0.0005
    dropout: float = 0.08
    patience: int = 12
    seed: int = 20260601
    ice_hits: int = 3


@dataclass
class PolicyDataset:
    """Flat candidate-level dataset plus state grouping metadata."""

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
    """Small one-hidden-layer MLP for ranking candidate moves."""

    def __init__(self, input_size: int, hidden_units: int = 48, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, math.sqrt(2.0 / input_size), size=(input_size, hidden_units)).astype(np.float32)
        self.b1 = np.zeros(hidden_units, dtype=np.float32)
        self.w2 = rng.normal(0.0, math.sqrt(2.0 / hidden_units), size=(hidden_units, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
        self.mean = np.zeros(input_size, dtype=np.float32)
        self.std = np.ones(input_size, dtype=np.float32)

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def _forward(self, x: np.ndarray, dropout: float = 0.0, rng=None):
        z1 = x @ self.w1 + self.b1
        h = np.maximum(z1, 0.0)
        mask = None
        if dropout > 0.0 and rng is not None:
            keep = 1.0 - dropout
            mask = (rng.random(h.shape) < keep).astype(np.float32) / keep
            h = h * mask
        pred = (h @ self.w2 + self.b2).reshape(-1)
        return pred, z1, h, mask

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        pred, _, _, _ = self._forward(self._standardize(x))
        return pred

    def fit(self, train: PolicyDataset, val: PolicyDataset, config: TrainingConfig):
        self.mean = train.x.mean(axis=0).astype(np.float32)
        self.std = np.maximum(train.x.std(axis=0), 1e-4).astype(np.float32)
        x_train = self._standardize(train.x)
        x_val = self._standardize(val.x)
        y_train = train.y.astype(np.float32)
        y_val = val.y.astype(np.float32)
        rng = np.random.default_rng(config.seed)
        best = None
        best_val = float("inf")
        stale_epochs = 0
        history = []

        adam = {name: [np.zeros_like(param), np.zeros_like(param)] for name, param in self._params().items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0

        for epoch in range(1, config.epochs + 1):
            order = rng.permutation(x_train.shape[0])
            x_epoch = x_train[order]
            y_epoch = y_train[order]
            pred, z1, h, mask = self._forward(x_epoch, dropout=config.dropout, rng=rng)
            err = pred - y_epoch
            train_loss = float(np.mean(err ** 2))

            grad_pred = (2.0 / max(1, x_epoch.shape[0])) * err.reshape(-1, 1)
            grad_w2 = h.T @ grad_pred + config.l2 * self.w2
            grad_b2 = grad_pred.sum(axis=0)
            grad_h = grad_pred @ self.w2.T
            if mask is not None:
                grad_h *= mask
            grad_z1 = grad_h * (z1 > 0.0)
            grad_w1 = x_epoch.T @ grad_z1 + config.l2 * self.w1
            grad_b1 = grad_z1.sum(axis=0)

            grads = {"w1": grad_w1, "b1": grad_b1, "w2": grad_w2, "b2": grad_b2}
            step += 1
            for name, param in self._params().items():
                m, v = adam[name]
                m[:] = beta1 * m + (1.0 - beta1) * grads[name]
                v[:] = beta2 * v + (1.0 - beta2) * (grads[name] ** 2)
                param -= config.learning_rate * (m / (1.0 - beta1 ** step)) / (np.sqrt(v / (1.0 - beta2 ** step)) + eps)

            val_pred = self._forward(x_val)[0]
            val_loss = float(np.mean((val_pred - y_val) ** 2))
            val_top1 = top1_agreement(val_pred, val.y, val.state_ids)
            history.append({"epoch": epoch, "trainLoss": train_loss, "valLoss": val_loss, "valTop1Agreement": val_top1})

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best = self._snapshot()
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= config.patience:
                    break

        if best is not None:
            self._restore(best)
        return {
            "epochsRun": len(history),
            "bestValLoss": best_val,
            "history": history,
            "trainTop1Agreement": top1_agreement(self.predict(train.x), train.y, train.state_ids),
            "valTop1Agreement": top1_agreement(self.predict(val.x), val.y, val.state_ids),
        }

    def _params(self):
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    def _snapshot(self):
        return {name: value.copy() for name, value in {**self._params(), "mean": self.mean, "std": self.std}.items()}

    def _restore(self, snapshot):
        for name, value in snapshot.items():
            setattr(self, name, value.copy())

    def to_dict(self, metrics=None, config=None):
        return {
            "format": "aiponchik-neural-move-policy-v1",
            "featureNames": list(FEATURE_NAMES),
            "standardization": {"mean": self.mean.tolist(), "std": self.std.tolist()},
            "network": {
                "activation": "relu",
                "w1": self.w1.tolist(),
                "b1": self.b1.tolist(),
                "w2": self.w2.tolist(),
                "b2": self.b2.tolist(),
            },
            "trainingConfig": config.__dict__ if config else None,
            "metrics": metrics or {},
        }

    @classmethod
    def from_dict(cls, payload: Dict):
        if payload.get("featureNames") != list(FEATURE_NAMES):
            raise ValueError("Policy feature list does not match this code version")
        network = payload["network"]
        model = cls(len(FEATURE_NAMES), len(network["b1"]))
        model.w1 = np.asarray(network["w1"], dtype=np.float32)
        model.b1 = np.asarray(network["b1"], dtype=np.float32)
        model.w2 = np.asarray(network["w2"], dtype=np.float32)
        model.b2 = np.asarray(network["b2"], dtype=np.float32)
        model.mean = np.asarray(payload["standardization"]["mean"], dtype=np.float32)
        model.std = np.asarray(payload["standardization"]["std"], dtype=np.float32)
        return model

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
    """Measure how often the model picks the same best candidate as labels."""
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


def train_policy(config: TrainingConfig, etalon_dir="etalon_images"):
    builder = PolicyTrainingDatasetBuilder(
        etalon_dir=etalon_dir,
        seed=config.seed,
        ice_hits=config.ice_hits,
        candidates_per_state=config.candidates_per_state,
    )
    train = builder.build(config.train_games, seed_offset=0)
    val = builder.build(config.val_games, seed_offset=100_000)
    model = NeuralMovePolicy(len(FEATURE_NAMES), hidden_units=config.hidden_units, seed=config.seed)
    metrics = model.fit(train, val, config)
    metrics.update({
        "trainSamples": train.size,
        "valSamples": val.size,
        "trainStates": train.state_count,
        "valStates": val.state_count,
    })
    return model, metrics


def select_policy_move(model: NeuralMovePolicy, encoder: MoveFeatureEncoder, state: Dict, candidates: Sequence[MoveCandidate], blend_heuristic: float = 0.0):
    """Pick a move by neural score, optionally blended with normalized heuristic score."""
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
    """Play simulated games with the trained policy and return win-rate metrics."""
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
