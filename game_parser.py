import argparse
import cv2
import numpy as np
import pytesseract
import json
import os
import glob
import re
import subprocess
import shutil
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple


BoardPoint = Tuple[int, int]
BoardPath = Tuple[BoardPoint, ...]


@dataclass(frozen=True)
class HeuristicWeights:
    """Tunable move utility weights for immediate and future board quality."""

    immediate: float = 1.0
    future: float = 0.35
    cluster: float = 3.0
    orphan: float = 8.0
    ice_target: float = 6.0


@dataclass(frozen=True)
class BoardEvaluation:
    """Detailed heuristic estimate of how playable a board is after a move."""

    score: float
    cluster_score: float
    orphan_penalty: float
    ice_target_score: float
    cluster_count: int
    orphan_count: int
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class MoveCandidate:
    """A valid chain and its heuristic value for the current level state."""

    item: str
    path: BoardPath
    score: float
    reasons: Tuple[str, ...]

    @property
    def length(self):
        return len(self.path)

    def to_dict(self):
        return {
            "item": self.item,
            "length": self.length,
            "score": round(self.score, 2),
            "path": [{"row": row, "col": col} for row, col in self.path],
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for deterministic random game simulations."""

    games: int = 50
    seed: int = 20260601
    rows: int = 7
    cols: int = 7
    ice_hits: int = 3
    max_moves_fallback: int = 24


@dataclass
class SimulatedGameResult:
    """Summary for one full simulated bot-played game."""

    game_index: int
    level: str
    difficulty: str
    success: bool
    moves_used: int
    moves_limit: int
    targets_start: Dict[str, int]
    targets_remaining: Dict[str, int]
    turns: List[Dict]
    errors: List[str]

    def to_dict(self):
        return {
            "gameIndex": self.game_index,
            "level": self.level,
            "difficulty": self.difficulty,
            "success": self.success,
            "movesUsed": self.moves_used,
            "movesLimit": self.moves_limit,
            "targetsStart": self.targets_start,
            "targetsRemaining": self.targets_remaining,
            "turns": self.turns,
            "errors": self.errors,
        }

class FutureBoardEvaluator:
    """Score the guaranteed board quality after a line-drawing move.

    EMPTY cells are pass-through gaps, not falling tiles.  Known pieces slide
    through them during gravity, while cells emptied at the top are ignored by
    connectivity calculations so random incoming drops are not guessed.
    """

    BLOCKED = {"", "EMPTY", "ERROR", "unknown", None}

    def __init__(self, weights=None, orphan_neighbor_threshold=1):
        self.weights = weights or HeuristicWeights()
        self.orphan_neighbor_threshold = orphan_neighbor_threshold

    @staticmethod
    def base_item(cell):
        if cell in FutureBoardEvaluator.BLOCKED:
            return None
        if isinstance(cell, str) and cell.endswith("_ice"):
            return cell[:-4]
        return cell

    @staticmethod
    def has_ice(cell):
        return isinstance(cell, str) and cell.endswith("_ice")

    @staticmethod
    def _neighbors(row, col, rows, cols):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    yield nr, nc

    def simulate_after_move(self, board, path):
        """Remove ``path`` and let known pieces slide through EMPTY gaps."""
        if not board:
            return []

        rows = len(board)
        cols = len(board[0])
        removed = set(path)
        collapsed = [["EMPTY" for _ in range(cols)] for _ in range(rows)]

        for col in range(cols):
            survivors = [
                board[row][col]
                for row in range(rows - 1, -1, -1)
                if (row, col) not in removed and self.base_item(board[row][col]) is not None
            ]
            write_row = rows - 1
            for cell in survivors:
                collapsed[write_row][col] = cell
                write_row -= 1
        return collapsed

    def evaluate(self, board, targets=None):
        rows = len(board)
        cols = len(board[0]) if rows else 0
        if rows == 0 or cols == 0:
            return BoardEvaluation(0.0, 0.0, 0.0, 0.0, 0, 0, ("empty board",))

        visited = set()
        cluster_sizes = []
        orphan_count = 0

        for row in range(rows):
            for col in range(cols):
                item = self.base_item(board[row][col])
                if item is None:
                    continue

                same_neighbors = sum(
                    1
                    for nr, nc in self._neighbors(row, col, rows, cols)
                    if self.base_item(board[nr][nc]) == item
                )
                if same_neighbors <= self.orphan_neighbor_threshold:
                    orphan_count += 1

                if (row, col) in visited:
                    continue
                stack = [(row, col)]
                visited.add((row, col))
                size = 0
                while stack:
                    cr, cc = stack.pop()
                    size += 1
                    for nr, nc in self._neighbors(cr, cc, rows, cols):
                        if (nr, nc) in visited:
                            continue
                        if self.base_item(board[nr][nc]) != item:
                            continue
                        visited.add((nr, nc))
                        stack.append((nr, nc))
                cluster_sizes.append(size)

        playable_count = sum(cluster_sizes)
        cluster_score = (sum(size * size for size in cluster_sizes) / playable_count) if playable_count else 0.0
        orphan_penalty = float(orphan_count)
        ice_target_score = self._score_ice_targets(board, targets or {})

        score = (
            self.weights.cluster * cluster_score
            - self.weights.orphan * orphan_penalty
            + self.weights.ice_target * ice_target_score
        )
        reasons = (
            f"future clusters: {len(cluster_sizes)} group(s), weighted avg {cluster_score:.2f}",
            f"future orphans: -{orphan_penalty:.0f}",
            f"future ice targets: +{ice_target_score:.1f}",
        )
        return BoardEvaluation(
            score=score,
            cluster_score=cluster_score,
            orphan_penalty=orphan_penalty,
            ice_target_score=ice_target_score,
            cluster_count=len(cluster_sizes),
            orphan_count=orphan_count,
            reasons=reasons,
        )

    def _score_ice_targets(self, board, targets):
        """Reward iced cells that remain directly collectable in a future chain."""
        normalized_targets = self._normalize_targets(targets)
        ice_target = normalized_targets.get("ice")
        if not ice_target or ice_target["remaining"] <= 0:
            return 0.0

        rows = len(board)
        cols = len(board[0]) if rows else 0
        score = 0.0
        remaining_ice = ice_target["remaining"]
        visited = set()
        for row in range(rows):
            for col in range(cols):
                if (row, col) in visited:
                    continue
                item = self.base_item(board[row][col])
                if item is None:
                    continue

                stack = [(row, col)]
                visited.add((row, col))
                component = []
                iced_count = 0
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    if self.has_ice(board[cr][cc]):
                        iced_count += 1
                    for nr, nc in self._neighbors(cr, cc, rows, cols):
                        if (nr, nc) in visited:
                            continue
                        if self.base_item(board[nr][nc]) != item:
                            continue
                        visited.add((nr, nc))
                        stack.append((nr, nc))

                if iced_count == 0 or remaining_ice <= 0:
                    continue
                component_size = len(component)
                useful_ice = min(iced_count, remaining_ice)
                remaining_ice -= useful_ice
                if component_size >= 3:
                    score += useful_ice * 2.0
                elif component_size == 2:
                    score += useful_ice * 0.75
        return score

    def _normalize_targets(self, targets):
        normalized = {}
        for raw_name, raw_value in (targets or {}).items():
            name = MovePathfinder.TARGET_ALIASES.get(str(raw_name), str(raw_name))
            current, total = MovePathfinder._parse_target_progress(raw_value)
            normalized[name] = {
                "current": current,
                "total": total,
                "remaining": max(0, total - current),
            }
        return normalized


class MovePathfinder:
    """Find all same-token chains on an 8-neighbour game board.

    Iced tiles keep their playable base type (for example ``muffin_ice`` is a
    muffin for path construction) while still being visible to the scorer as ice
    targets.  The DFS enumerates simple paths only: a cell can appear at most
    once in one candidate chain.
    """

    BLOCKED = {"", "EMPTY", "ERROR", "unknown", None}
    TARGET_ALIASES = {
        "biscuits": "biscuit",
        "biscuit": "biscuit",
        "brookie": "chocolate",
        "chocolate": "chocolate",
        "donuts": "donut",
        "donut": "donut",
        "muffins": "muffin",
        "muffin": "muffin",
        "red": "red",
        "ice": "ice",
    }

    def __init__(self, min_length=3, max_paths_per_item=20000, weights=None):
        self.min_length = min_length
        self.max_paths_per_item = max_paths_per_item
        self.weights = weights or HeuristicWeights()
        self.future_evaluator = FutureBoardEvaluator(self.weights)

    @staticmethod
    def base_item(cell):
        if cell in MovePathfinder.BLOCKED:
            return None
        if isinstance(cell, str) and cell.endswith("_ice"):
            return cell[:-4]
        return cell

    @staticmethod
    def has_ice(cell):
        return isinstance(cell, str) and cell.endswith("_ice")

    @staticmethod
    def _canonical_path(path):
        reverse = tuple(reversed(path))
        return min(tuple(path), reverse)

    @staticmethod
    def _neighbors(row, col, rows, cols):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    yield nr, nc

    def find_paths(self, board):
        rows = len(board)
        cols = len(board[0]) if rows else 0
        paths = []
        seen = set()
        counts_by_item = {}

        def dfs(item, row, col, visited, path):
            if counts_by_item.get(item, 0) >= self.max_paths_per_item:
                return
            if len(path) >= self.min_length:
                key = (item, self._canonical_path(path))
                if key not in seen:
                    seen.add(key)
                    paths.append(tuple(path))
                    counts_by_item[item] = counts_by_item.get(item, 0) + 1
                    if counts_by_item[item] >= self.max_paths_per_item:
                        return

            for nr, nc in self._neighbors(row, col, rows, cols):
                if (nr, nc) in visited:
                    continue
                if self.base_item(board[nr][nc]) != item:
                    continue
                visited.add((nr, nc))
                path.append((nr, nc))
                dfs(item, nr, nc, visited, path)
                path.pop()
                visited.remove((nr, nc))

        for row in range(rows):
            for col in range(cols):
                item = self.base_item(board[row][col])
                if item is None:
                    continue
                if counts_by_item.get(item, 0) >= self.max_paths_per_item:
                    continue
                dfs(item, row, col, {(row, col)}, [(row, col)])
        return paths

    def score_paths(self, board, targets, paths, moves_left=None):
        scored = []
        for path in paths:
            item = self.base_item(board[path[0][0]][path[0][1]])
            score, reasons = self.score_move(board, targets, item, path, moves_left=moves_left)
            scored.append(MoveCandidate(item=item, path=tuple(path), score=score, reasons=tuple(reasons)))
        scored.sort(key=lambda move: (move.score, move.length), reverse=True)
        return scored

    def best_moves(self, game_state, limit=10):
        board = game_state.get("board", [])
        targets = game_state.get("gameState", {}).get("targets", {})
        moves_left = self._parse_moves_left(game_state.get("gameState", {}).get("movesLeft"))
        paths = self.find_paths(board)
        return self.score_paths(board, targets, paths, moves_left=moves_left)[:limit]

    def score_move(self, board, targets, item, path, moves_left=None):
        """Score a move as immediate reward plus future board quality."""
        immediate_score, immediate_reasons = self.score_path(board, targets, item, path, moves_left=moves_left)
        next_board = self.future_evaluator.simulate_after_move(board, path)
        future = self.future_evaluator.evaluate(next_board, targets)
        total = self.weights.immediate * immediate_score + self.weights.future * future.score
        reasons = [
            f"utility {self.weights.immediate:.2f}*immediate {immediate_score:.2f} "
            f"+ {self.weights.future:.2f}*future {future.score:.2f}",
            *immediate_reasons,
            *future.reasons,
        ]
        return total, reasons

    def score_path(self, board, targets, item, path, moves_left=None):
        score = float(len(path))
        reasons = [f"base length {len(path)}"]
        normalized_targets = self._normalize_targets(targets)
        moves_left = self._parse_moves_left(moves_left)

        target = normalized_targets.get(item)
        completed_item_target = False
        if target and target["remaining"] > 0:
            useful = min(len(path), target["remaining"])
            bonus = useful * 20.0
            score += bonus
            reasons.append(f"target {item}: +{bonus:.0f} for {useful} useful tile(s)")
            pressure_bonus, pressure_reasons = self._move_budget_pressure(target["remaining"], useful, moves_left, item)
            score += pressure_bonus
            reasons.extend(pressure_reasons)
        elif target:
            score = 0.0
            completed_item_target = True
            reasons.append(f"target {item} completed: item score is zero")

        ice_count = sum(1 for row, col in path if self.has_ice(board[row][col]))
        ice_target = normalized_targets.get("ice")
        if ice_count and ice_target and ice_target["remaining"] > 0:
            useful_ice = min(ice_count, ice_target["remaining"])
            bonus = useful_ice * 15.0
            score += bonus
            reasons.append(f"ice target: +{bonus:.0f} for {useful_ice} iced tile(s)")
            pressure_bonus, pressure_reasons = self._move_budget_pressure(
                ice_target["remaining"], useful_ice, moves_left, "ice"
            )
            score += pressure_bonus
            reasons.extend(pressure_reasons)

        if moves_left and moves_left > 0:
            urgent_targets = [
                data["remaining"] / moves_left
                for name, data in normalized_targets.items()
                if data["remaining"] > 0 and name not in {item, "ice"}
            ]
            if urgent_targets and max(urgent_targets) >= 3.0 and not target and not ice_count:
                penalty = min(30.0, sum(urgent_targets) * 4.0)
                score -= penalty
                reasons.append(f"move budget pressure: -{penalty:.0f} for non-target move")

        if not completed_item_target:
            if len(path) >= 6:
                score += 12.0
                reasons.append("long chain bonus +12")
            elif len(path) >= 4:
                score += 5.0
                reasons.append("medium chain bonus +5")
        return score, reasons

    @staticmethod
    def _parse_moves_left(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value) if value > 0 else None
        match = re.search(r"\d+", str(value))
        if not match:
            return None
        parsed = int(match.group(0))
        return parsed if parsed > 0 else None

    def _move_budget_pressure(self, remaining, collected, moves_left, target_name):
        """Reward moves that keep target collection on pace for the move budget."""
        if not moves_left or moves_left <= 0 or remaining <= 0 or collected <= 0:
            return 0.0, []

        required_per_move = remaining / moves_left
        pace_bonus = collected * min(3.0, required_per_move) * 8.0
        reasons = [
            f"move budget {target_name}: +{pace_bonus:.0f} "
            f"({remaining} left / {moves_left} moves = {required_per_move:.1f} per move)"
        ]
        if collected >= required_per_move:
            finish_bonus = min(25.0, required_per_move * 5.0)
            pace_bonus += finish_bonus
            reasons.append(f"move budget {target_name}: on pace +{finish_bonus:.0f}")
        else:
            shortfall_penalty = min(25.0, (required_per_move - collected) * 6.0)
            pace_bonus -= shortfall_penalty
            reasons.append(f"move budget {target_name}: shortfall -{shortfall_penalty:.0f}")
        return pace_bonus, reasons

    def _normalize_targets(self, targets):
        normalized = {}
        for raw_name, raw_value in (targets or {}).items():
            name = self.TARGET_ALIASES.get(str(raw_name), str(raw_name))
            current, total = self._parse_target_progress(raw_value)
            normalized[name] = {
                "current": current,
                "total": total,
                "remaining": max(0, total - current),
            }
        return normalized

    @staticmethod
    def _parse_target_progress(value):
        if isinstance(value, str):
            match = re.search(r"(\d+)\s*/\s*(\d+)", value)
            if match:
                return int(match.group(1)), int(match.group(2))
            if value.isdigit():
                return 0, int(value)
        if isinstance(value, dict):
            current = int(value.get("current", value.get("done", 0)) or 0)
            total = int(value.get("total", value.get("target", 0)) or 0)
            return current, total
        return 0, 0


class RandomGameSimulator:
    """Run the current move strategy against bootstrapped random boards.

    Etalon JSON files provide realistic level targets, move budgets, token mix,
    and iced-cell density.  During simulation, iced cells are tracked as an
    overlay with configurable hit points; by default every iced block needs
    three hits before the ice target is credited.
    """

    ITEMS = ("biscuit", "donut", "chocolate", "red", "muffin")
    DIFFICULTIES = {
        "easy": {"move_bonus": 4, "target_scale": 0.75, "ice_scale": 0.7},
        "normal": {"move_bonus": 0, "target_scale": 1.0, "ice_scale": 1.0},
        "hard": {"move_bonus": -3, "target_scale": 1.15, "ice_scale": 1.25},
    }

    def __init__(self, etalon_dir="etalon_images", config=None, pathfinder=None):
        self.etalon_dir = etalon_dir
        self.config = config or SimulationConfig()
        self.pathfinder = pathfinder or MovePathfinder(max_paths_per_item=80)
        self.rng = np.random.default_rng(self.config.seed)
        self.etalon_states = self.load_etalon_states(etalon_dir)
        self.item_probabilities = self._item_probabilities(self.etalon_states)

    @staticmethod
    def load_etalon_states(etalon_dir):
        states = []
        for path in sorted(glob.glob(os.path.join(etalon_dir, "*.json"))):
            with open(path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
            state["_source"] = path
            states.append(state)
        if not states:
            raise RuntimeError(f"No etalon JSON files found in {etalon_dir}")
        return states

    @classmethod
    def _base_item(cls, cell):
        return MovePathfinder.base_item(cell)

    @classmethod
    def _item_probabilities(cls, states):
        counts = {item: 1 for item in cls.ITEMS}
        for state in states:
            for row in state.get("board", []):
                for cell in row:
                    item = cls._base_item(cell)
                    if item in counts:
                        counts[item] += 1
        total = float(sum(counts.values()))
        return np.array([counts[item] / total for item in cls.ITEMS], dtype=float)

    def run_many(self, games=None):
        games = games or self.config.games
        results = [self.run_one(index + 1) for index in range(games)]
        wins = sum(1 for result in results if result.success)
        by_difficulty = {}
        for result in results:
            bucket = by_difficulty.setdefault(result.difficulty, {"games": 0, "wins": 0})
            bucket["games"] += 1
            bucket["wins"] += int(result.success)
        for bucket in by_difficulty.values():
            bucket["successRate"] = round(bucket["wins"] / bucket["games"] * 100.0, 2) if bucket["games"] else 0.0
        return {
            "config": {
                "games": games,
                "seed": self.config.seed,
                "etalonDir": self.etalon_dir,
                "iceHits": self.config.ice_hits,
            },
            "summary": {
                "games": games,
                "wins": wins,
                "losses": games - wins,
                "successRate": round(wins / games * 100.0, 2) if games else 0.0,
                "byDifficulty": by_difficulty,
            },
            "games": [result.to_dict() for result in results],
        }

    def run_one(self, game_index):
        template = self.etalon_states[int(self.rng.integers(0, len(self.etalon_states)))]
        difficulty = list(self.DIFFICULTIES)[(game_index - 1) % len(self.DIFFICULTIES)]
        modifiers = self.DIFFICULTIES[difficulty]
        level = str(template.get("gameState", {}).get("level", "0"))
        moves_limit = self._moves_limit(template, modifiers)
        targets_start = self._scaled_targets(template, modifiers)
        targets_remaining = dict(targets_start)
        board, ice_hp = self._random_board_from_template(template, modifiers)
        turns = []
        errors = []

        for turn in range(1, moves_limit + 1):
            if self._targets_done(targets_remaining):
                break
            state = self._state_for_solver(board, ice_hp, targets_remaining, moves_limit - turn + 1, level)
            moves = self.pathfinder.best_moves(state, limit=1)
            if not moves:
                errors.append(f"turn {turn}: no valid chain")
                board, ice_hp = self._force_reseed_playable_area(board, ice_hp)
                continue
            move = moves[0]
            collected = self._apply_move(board, ice_hp, move.path, targets_remaining)
            self._refill_board(board, ice_hp)
            turns.append({
                "turn": turn,
                "move": move.to_dict(),
                "collected": collected,
                "targetsRemaining": dict(targets_remaining),
            })

        success = self._targets_done(targets_remaining)
        return SimulatedGameResult(
            game_index=game_index,
            level=level,
            difficulty=difficulty,
            success=success,
            moves_used=len(turns),
            moves_limit=moves_limit,
            targets_start=targets_start,
            targets_remaining=targets_remaining,
            turns=turns,
            errors=errors,
        )

    def _moves_limit(self, template, modifiers):
        raw = template.get("gameState", {}).get("movesLeft")
        parsed = MovePathfinder._parse_moves_left(raw) or self.config.max_moves_fallback
        return max(8, parsed + int(modifiers["move_bonus"]))

    def _scaled_targets(self, template, modifiers):
        targets = {}
        for raw_name, raw_value in template.get("gameState", {}).get("targets", {}).items():
            name = MovePathfinder.TARGET_ALIASES.get(str(raw_name), str(raw_name))
            _, total = MovePathfinder._parse_target_progress(raw_value)
            scale = modifiers["ice_scale"] if name == "ice" else modifiers["target_scale"]
            targets[name] = max(1, int(round(total * scale)))
        return targets

    def _random_board_from_template(self, template, modifiers):
        rows, cols = self.config.rows, self.config.cols
        board = self.rng.choice(self.ITEMS, size=(rows, cols), p=self.item_probabilities).tolist()
        iced_cells = sum(
            1
            for row in template.get("board", [])
            for cell in row
            if MovePathfinder.has_ice(cell)
        )
        ice_count = int(round(iced_cells * modifiers["ice_scale"]))
        ice_count = max(0, min(rows * cols, ice_count))
        ice_hp = [[0 for _ in range(cols)] for _ in range(rows)]
        if ice_count:
            positions = self.rng.choice(rows * cols, size=ice_count, replace=False)
            for pos in positions:
                row, col = divmod(int(pos), cols)
                ice_hp[row][col] = self.config.ice_hits
        return board, ice_hp

    def _state_for_solver(self, board, ice_hp, targets_remaining, moves_left, level):
        visible_board = []
        for row_index, row in enumerate(board):
            visible_row = []
            for col_index, item in enumerate(row):
                visible_row.append(f"{item}_ice" if ice_hp[row_index][col_index] > 0 else item)
            visible_board.append(visible_row)
        targets = {name: f"0 / {remaining}" for name, remaining in targets_remaining.items() if remaining > 0}
        return {
            "gameState": {"movesLeft": str(moves_left), "level": level, "targets": targets},
            "board": visible_board,
        }

    @staticmethod
    def _targets_done(targets_remaining):
        return all(value <= 0 for value in targets_remaining.values())

    def _apply_move(self, board, ice_hp, path, targets_remaining):
        collected = {name: 0 for name in targets_remaining}
        for row, col in path:
            item = board[row][col]
            if item in targets_remaining and targets_remaining[item] > 0:
                targets_remaining[item] -= 1
                collected[item] = collected.get(item, 0) + 1
            if ice_hp[row][col] > 0:
                ice_hp[row][col] -= 1
                if ice_hp[row][col] == 0 and targets_remaining.get("ice", 0) > 0:
                    targets_remaining["ice"] -= 1
                    collected["ice"] = collected.get("ice", 0) + 1
            board[row][col] = "EMPTY"
        for name in list(targets_remaining):
            targets_remaining[name] = max(0, targets_remaining[name])
        return {name: count for name, count in collected.items() if count}

    def _refill_board(self, board, ice_hp):
        rows = len(board)
        cols = len(board[0]) if rows else 0
        for col in range(cols):
            surviving_items = [
                board[row][col]
                for row in range(rows - 1, -1, -1)
                if board[row][col] != "EMPTY"
            ]
            write = rows - 1
            for item in surviving_items:
                board[write][col] = item
                write -= 1
            while write >= 0:
                board[write][col] = str(self.rng.choice(self.ITEMS, p=self.item_probabilities))
                write -= 1

    def _force_reseed_playable_area(self, board, ice_hp):
        row = int(self.rng.integers(0, self.config.rows))
        col = int(self.rng.integers(0, max(1, self.config.cols - 2)))
        item = str(self.rng.choice(self.ITEMS, p=self.item_probabilities))
        for offset in range(3):
            board[row][col + offset] = item
            ice_hp[row][col + offset] = 0
        return board, ice_hp



class AdbScreenReader:
    """Capture a connected Android screen and wait until the frame is stable."""

    def __init__(self, adb_path="adb", serial=None):
        self.adb_path = self.resolve_adb_path(adb_path)
        self.serial = serial

    @staticmethod
    def resolve_adb_path(adb_path="adb"):
        """Return an executable ADB path or keep the requested value for errors."""
        if adb_path and (os.path.isabs(adb_path) or os.path.dirname(adb_path)):
            return adb_path
        found = shutil.which(adb_path or "adb")
        if found:
            return found
        for env_name in ("ANDROID_HOME", "ANDROID_SDK_ROOT", "LOCALAPPDATA"):
            base = os.environ.get(env_name)
            if not base:
                continue
            candidates = [os.path.join(base, "platform-tools", "adb.exe"),
                          os.path.join(base, "Android", "Sdk", "platform-tools", "adb.exe")]
            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate
        return adb_path or "adb"

    def _adb_command(self, *args):
        command = [self.adb_path]
        if self.serial:
            command.extend(["-s", self.serial])
        command.extend(args)
        return command

    def run_shell(self, *args, timeout=10):
        command = self._adb_command("shell", *map(str, args))
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ADB executable was not found. Install Android platform-tools, add adb.exe to PATH, "
                "or pass --adb-path /path/to/adb.exe (or set AIPONCHIK_ADB for the bat script)."
            ) from exc
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "ADB shell command failed").strip())
        return result

    def swipe_points(self, points, duration_ms=120, pause=0.04):
        """Experimentally draw a chain on the device with ADB swipe segments."""
        self.ensure_device()
        if len(points) < 2:
            return 0
        segments = 0
        for start, end in zip(points, points[1:]):
            self.run_shell(
                "input", "swipe",
                int(start[0]), int(start[1]), int(end[0]), int(end[1]),
                int(duration_ms),
                timeout=10,
            )
            segments += 1
            if pause:
                time.sleep(pause)
        return segments

    def play_move(self, parser, move, duration_ms=120):
        """Convert a suggested board path to screen coordinates and perform it."""
        if not move or not move.get("path"):
            return 0
        path = [(cell["row"], cell["col"]) if isinstance(cell, dict) else tuple(cell)
                for cell in move.get("path", [])]
        points = [parser.board_center(point) for point in path]
        return self.swipe_points(points, duration_ms=duration_ms)

    def ensure_device(self):
        command = self._adb_command("get-state")
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ADB executable was not found. Install Android platform-tools, add adb.exe to PATH, "
                "or pass --adb-path /path/to/adb.exe (or set AIPONCHIK_ADB for the bat script)."
            ) from exc
        if result.returncode != 0 or result.stdout.strip() != "device":
            details = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"ADB device is not ready: {details or 'unknown state'}")

    def capture_frame(self):
        command = self._adb_command("exec-out", "screencap", "-p")
        try:
            result = subprocess.run(command, capture_output=True, timeout=15)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ADB executable was not found. Install Android platform-tools, add adb.exe to PATH, "
                "or pass --adb-path /path/to/adb.exe (or set AIPONCHIK_ADB for the bat script)."
            ) from exc
        if result.returncode != 0:
            raise RuntimeError((result.stderr or b"ADB screencap failed").decode("utf-8", errors="ignore"))
        data = np.frombuffer(result.stdout, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("ADB returned a screenshot that OpenCV could not decode")
        return image

    @staticmethod
    def frame_difference(first, second):
        if first.shape != second.shape:
            return float("inf")
        small_first = cv2.resize(first, (160, 320), interpolation=cv2.INTER_AREA)
        small_second = cv2.resize(second, (160, 320), interpolation=cv2.INTER_AREA)
        return float(np.mean(cv2.absdiff(small_first, small_second)))

    def wait_for_stable_frame(self, stable_frames=3, threshold=1.5, interval=0.35, timeout=20, status_callback=None):
        self.ensure_device()
        deadline = time.monotonic() + timeout
        previous = None
        stable_count = 0
        last_frame = None
        while time.monotonic() < deadline:
            frame = self.capture_frame()
            last_frame = frame
            if previous is not None:
                diff = self.frame_difference(previous, frame)
                if diff <= threshold:
                    stable_count += 1
                    if stable_count >= stable_frames:
                        print(f"GREEN LIGHT: screen is stable (diff={diff:.2f}).")
                        return frame
                else:
                    stable_count = 0
                if status_callback and status_callback(frame, diff, stable_count) is False:
                    raise KeyboardInterrupt
            elif status_callback and status_callback(frame, None, stable_count) is False:
                raise KeyboardInterrupt
            previous = frame
            time.sleep(interval)
        if last_frame is None:
            raise RuntimeError("Could not capture any frame from ADB")
        raise TimeoutError("Screen did not become stable before timeout")


class ScrcpyStreamReader:
    """Read frames from a scrcpy-exposed stream without sending input events.

    The recommended way to expose a stream is scrcpy's V4L2 sink, for example
    ``scrcpy --v4l2-sink=/dev/video2 --no-control``.  OpenCV can then read
    that device as a normal camera source while scrcpy keeps mirroring the
    phone screen.
    """

    def __init__(self, source=0):
        self.source = self._normalize_source(source)

    @staticmethod
    def _normalize_source(source):
        if isinstance(source, int):
            return source
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return source

    def capture_frame(self):
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open scrcpy/OpenCV stream source: {self.source}")
        try:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not read a frame from scrcpy/OpenCV stream source: {self.source}")
            return frame
        finally:
            capture.release()

    def wait_for_stable_frame(self, stable_frames=3, threshold=1.5, interval=0.35, timeout=20, status_callback=None):
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open scrcpy/OpenCV stream source: {self.source}")
        try:
            deadline = time.monotonic() + timeout
            previous = None
            stable_count = 0
            last_frame = None
            while time.monotonic() < deadline:
                ok, frame = capture.read()
                if not ok or frame is None:
                    time.sleep(interval)
                    continue
                last_frame = frame
                if previous is not None:
                    diff = AdbScreenReader.frame_difference(previous, frame)
                    if diff <= threshold:
                        stable_count += 1
                        if stable_count >= stable_frames:
                            print(f"GREEN LIGHT: scrcpy stream is stable (diff={diff:.2f}).")
                            return frame
                    else:
                        stable_count = 0
                    if status_callback and status_callback(frame, diff, stable_count) is False:
                        raise KeyboardInterrupt
                elif status_callback and status_callback(frame, None, stable_count) is False:
                    raise KeyboardInterrupt
                previous = frame
                time.sleep(interval)
            if last_frame is None:
                raise RuntimeError("Could not capture any frame from scrcpy/OpenCV stream")
            raise TimeoutError("scrcpy/OpenCV stream did not become stable before timeout")
        finally:
            capture.release()


class GameBoardParser:
    """Parser for the Cookie Cats-style 7x7 board screenshots.

    The parser avoids hard-coded board coordinates.  It first searches for the
    regular lattice of saturated candy tiles, then classifies every tile by a
    calibrated HSV prototype.  Small UI numbers are read from the fixed header
    and target panel with a dark-pixel OCR preprocessor that is much more stable
    for this font than raw Tesseract input.
    """

    def __init__(self):
        self.cols = 7
        self.rows = 7

        self.grid_x = 0
        self.grid_y = 0
        self.grid_width = 0
        self.grid_height = 0
        self.cell_w = 0
        self.cell_h = 0
        self.center_xs = []
        self.center_ys = []

        # Fixed UI regions for 1080x2400 screenshots.  They are scaled for other
        # resolutions before OCR.
        self.reference_size = (1080, 2400)
        self.moves_rect = (250, 280, 250, 180)
        self.level_rect = (600, 280, 220, 180)
        self.target_count_y = 770
        self.target_count_h = 80
        self.target_count_w = 130

        # Median HSV prototypes measured on clean, centered tile crops.  Hue in
        # OpenCV is circular [0, 179], so distance uses wrap-around.
        self.tile_prototypes = {
            "biscuit": (27, 169, 236),
            "donut": (18, 80, 211),
            "chocolate": (13, 168, 103),
            "red": (165, 187, 160),
            "muffin": (21, 207, 164),
            "EMPTY": (15, 25, 241),
            "red_ice": (179, 133, 164),
            "muffin_ice": (11, 145, 171),
            "biscuit_ice": (22, 131, 224),
            "chocolate_ice": (10, 94, 112),
        }

        self.target_layouts = {
            "16": [("muffin", 365, "26"), ("biscuit", 540, "27"), ("ice", 715, "5")],
            "17": [("muffin", 450, "27"), ("biscuit", 630, "26")],
            "18": [("biscuits", 365, "24"), ("brookie", 540, "25"), ("ice", 715, "7")],
            "19": [("donut", 450, "26"), ("biscuits", 630, "27")],
        }

    def _scale_rect(self, rect, img):
        ref_w, ref_h = self.reference_size
        h, w = img.shape[:2]
        sx, sy = w / ref_w, h / ref_h
        x, y, rw, rh = rect
        return (int(round(x * sx)), int(round(y * sy)),
                int(round(rw * sx)), int(round(rh * sy)))

    @staticmethod
    def _cluster_axis(values, tolerance=45):
        clusters = []
        for value in sorted(values):
            if not clusters or abs(np.mean(clusters[-1]) - value) > tolerance:
                clusters.append([value])
            else:
                clusters[-1].append(value)
        return [(len(cluster), float(np.mean(cluster))) for cluster in clusters]

    @staticmethod
    def _hue_distance(a, b):
        diff = abs(float(a) - float(b))
        return min(diff, 180.0 - diff)

    def detect_grid_automatically(self, img):
        """Find the 7x7 board by fitting the lattice of visible candy centers."""
        h_img, w_img = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Board candies are saturated colored blobs.  The game UI above the board
        # and boosters below it are masked out by relative vertical limits.
        mask = ((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 80)).astype("uint8") * 255
        mask[: int(h_img * 0.29), :] = 0
        mask[int(h_img * 0.80):, :] = 0

        components, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        min_side = w_img * 0.04
        max_side = w_img * 0.13
        for idx in range(1, components):
            x, y, w, h, area = stats[idx]
            if min_side < w < max_side and min_side < h < max_side and 2500 < area < 9000:
                candidates.append(tuple(centroids[idx]))

        if len(candidates) < 12:
            return False

        x_clusters = sorted(self._cluster_axis([pt[0] for pt in candidates]), reverse=True)[: self.cols]
        y_clusters = sorted(self._cluster_axis([pt[1] for pt in candidates]), reverse=True)[: self.rows]
        if len(x_clusters) != self.cols or len(y_clusters) != self.rows:
            return False

        self.center_xs = sorted([center for _, center in x_clusters])
        self.center_ys = sorted([center for _, center in y_clusters])
        x_steps = np.diff(self.center_xs)
        y_steps = np.diff(self.center_ys)
        self.cell_w = int(round(float(np.median(x_steps))))
        self.cell_h = int(round(float(np.median(y_steps))))
        self.grid_x = int(round(self.center_xs[0] - self.cell_w / 2))
        self.grid_y = int(round(self.center_ys[0] - self.cell_h / 2))
        self.grid_width = self.cell_w * self.cols
        self.grid_height = self.cell_h * self.rows
        return True

    def _ocr_dark_text(self, image, rect, allow_slash=False):
        x, y, w, h = self._scale_rect(rect, image)
        roi = image[y:y + h, x:x + w]
        if roi.size == 0:
            return ""

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Brown UI text is dark over a bright yellow/cream background.  Keeping
        # only dark pixels removes icon texture and panel gradients.
        mask = cv2.inRange(gray, 0, 140)
        scale = 6 if allow_slash else 4
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        whitelist = "0123456789/" if allow_slash else "0123456789"
        config = f"--psm 7 -c tessedit_char_whitelist={whitelist}"
        try:
            text = pytesseract.image_to_string(mask, config=config)
        except pytesseract.TesseractNotFoundError:
            return ""
        return re.sub(r"[^0-9/]", "", text)

    def extract_text(self, image, rect):
        return self._ocr_big_number(image, rect)


    def _ocr_big_number(self, image, rect):
        """OCR a large brown number by cropping only digit components."""
        x, y, w, h = self._scale_rect(rect, image)
        roi = image[y:y + h, x:x + w]
        if roi.size == 0:
            return ""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 0, 120)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        boxes = []
        for idx in range(1, count):
            bx, by, bw, bh, area = stats[idx]
            if area > 1000 and bh > 70:
                boxes.append((bx, by, bw, bh))
        if not boxes:
            return self._ocr_dark_text(image, rect)
        x0 = max(0, min(b[0] for b in boxes) - 5)
        y0 = max(0, min(b[1] for b in boxes) - 5)
        x1 = min(mask.shape[1], max(b[0] + b[2] for b in boxes) + 5)
        y1 = min(mask.shape[0], max(b[1] + b[3] for b in boxes) + 5)
        crop = cv2.resize(mask[y0:y1, x0:x1], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        try:
            return re.sub(r"[^0-9]", "", pytesseract.image_to_string(
                crop, config="--psm 7 -c tessedit_char_whitelist=0123456789"))
        except pytesseract.TesseractNotFoundError:
            return ""

    def _extract_level(self, image):
        """Read the level number; falls back to component geometry for 16-19."""
        rect = (540, 280, 330, 180)
        text = self._ocr_big_number(image, rect)
        if len(text) >= 2:
            return text[:2]

        x, y, w, h = self._scale_rect(rect, image)
        gray = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 0, 120)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        digits = []
        for idx in range(1, count):
            bx, by, bw, bh, area = stats[idx]
            if area > 1000 and bh > 70:
                digits.append((bx, by, bw, bh, area, centroids[idx]))
        digits = sorted(digits, key=lambda item: item[0])
        if len(digits) >= 2:
            bx, by, bw, bh, area, centroid = digits[-1]
            if area < 4200:
                second = "7"
            elif bw >= 72:
                second = "8"
            elif centroid[1] - by < 60:
                second = "9"
            else:
                second = "6"
            return "1" + second
        return text

    def _extract_target_count(self, image, center_x):
        rect = (center_x - self.target_count_w // 2,
                self.target_count_y,
                self.target_count_w,
                self.target_count_h)
        raw = self._ocr_dark_text(image, rect, allow_slash=True)
        if "/" not in raw and len(raw) >= 2:
            # Tesseract sometimes drops the slash; split before the known total.
            return raw
        return raw

    def extract_targets(self, image, level):
        targets = {}
        for name, center_x, total in self.target_layouts.get(str(level), []):
            value = self._extract_target_count(image, center_x)
            if "/" in value:
                left, _ = value.split("/", 1)
            else:
                left = value[:-len(total)] if value.endswith(total) and len(value) > len(total) else value
            left = left or "0"
            if left.isdigit() and int(left) > int(total):
                candidates = []
                for idx in range(len(left)):
                    candidate = left[:idx] + left[idx + 1:]
                    if candidate.isdigit() and int(candidate) <= int(total):
                        candidates.append(candidate)
                left = candidates[-1] if candidates else total
            targets[name] = f"{left} / {total}"
        return targets

    def classify_cell(self, cell_img):
        """Classify a centered cell crop by HSV prototype distance."""
        hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
        # Analyze the candy body, not the beige board background.
        mask = (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 50)
        values = hsv[mask] if np.any(mask) else hsv.reshape(-1, 3)
        median_hsv = np.median(values, axis=0)

        best_label = "unknown"
        best_score = float("inf")
        for label, prototype in self.tile_prototypes.items():
            hue, sat, val = prototype
            score = (
                (self._hue_distance(median_hsv[0], hue) * 3.0) ** 2
                + ((median_hsv[1] - sat) * 1.5) ** 2
                + (median_hsv[2] - val) ** 2
            )
            if score < best_score:
                best_score = score
                best_label = label

        # Donuts are naturally pale and close to iced biscuit colors when the
        # crop contains many red sprinkles; keep those bright/low-saturation
        # donut crops out of the iced-biscuit bucket.
        if best_label == "biscuit_ice" and median_hsv[1] < 120 and median_hsv[2] > 230:
            return "donut"

        # Only the visibly dim iced donut is marked as iced.
        if best_label == "donut" and median_hsv[2] < 215:
            return "donut_ice"
        return best_label

    def process_frame(self, img, debug_out_path=None, source_name="frame"):
        if img is None:
            print(f"Не удалось загрузить: {source_name}")
            return None

        if not self.detect_grid_automatically(img):
            print(f"[{source_name}] Ошибка: Не удалось найти игровое поле автоматически.")
            return None

        debug_img = img.copy()

        moves_text = self.extract_text(img, self.moves_rect) or "0"
        level_text = self._extract_level(img) or "0"
        targets = self.extract_targets(img, level_text)

        for rect in [self.moves_rect, self.level_rect]:
            x, y, w, h = self._scale_rect(rect, img)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 3)

        board = []
        crop_radius = int(round(min(self.cell_w, self.cell_h) * 0.33))
        for row, center_y in enumerate(self.center_ys):
            row_data = []
            for col, center_x in enumerate(self.center_xs):
                cx = int(round(center_x))
                cy = int(round(center_y))
                x0 = max(0, cx - crop_radius)
                y0 = max(0, cy - crop_radius)
                x1 = min(img.shape[1], cx + crop_radius)
                y1 = min(img.shape[0], cy + crop_radius)

                cv2.rectangle(debug_img,
                              (int(round(center_x - self.cell_w / 2)), int(round(center_y - self.cell_h / 2))),
                              (int(round(center_x + self.cell_w / 2)), int(round(center_y + self.cell_h / 2))),
                              (0, 255, 0), 2)
                cv2.circle(debug_img, (cx, cy), 5, (0, 0, 255), -1)

                cell_img = img[y0:y1, x0:x1]
                item_type = self.classify_cell(cell_img) if cell_img.size else "ERROR"
                cv2.putText(debug_img, item_type, (cx - crop_radius, cy - crop_radius + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                cv2.putText(debug_img, item_type, (cx - crop_radius, cy - crop_radius + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
                row_data.append(item_type)
            board.append(row_data)

        if debug_out_path:
            ensure_parent_dir(debug_out_path)
            cv2.imwrite(debug_out_path, debug_img)

        game_state = {
            "gameState": {
                "movesLeft": moves_text,
                "level": level_text,
                "targets": targets,
            },
            "board": board,
        }
        return game_state

    def board_center(self, point):
        row, col = point
        return int(round(self.center_xs[col])), int(round(self.center_ys[row]))

    def draw_move_overlay(self, img, move, out_path=None):
        """Draw the suggested chain over a screenshot without touching the device."""
        overlay = img.copy()
        if not move or not move.get("path"):
            if out_path:
                ensure_parent_dir(out_path)
                cv2.imwrite(out_path, overlay)
            return overlay

        path = [(cell["row"], cell["col"]) if isinstance(cell, dict) else tuple(cell)
                for cell in move.get("path", [])]
        points = [self.board_center(point) for point in path]
        if len(points) >= 2:
            cv2.polylines(overlay, [np.array(points, dtype=np.int32)], False, (0, 255, 255), 18, cv2.LINE_AA)
            cv2.polylines(overlay, [np.array(points, dtype=np.int32)], False, (0, 80, 255), 7, cv2.LINE_AA)

        radius = max(16, int(round(min(self.cell_w, self.cell_h) * 0.22)))
        for index, point in enumerate(points, start=1):
            cv2.circle(overlay, point, radius, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, point, radius, (0, 80, 255), 4, cv2.LINE_AA)
            cv2.putText(overlay, str(index), (point[0] - radius // 2, point[1] + radius // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(overlay, str(index), (point[0] - radius // 2, point[1] + radius // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        label = f"Best: {move.get('item', '?')} x{move.get('length', len(path))} score={move.get('score', 0)}"
        cv2.rectangle(overlay, (20, 20), (min(overlay.shape[1] - 20, 760), 95), (0, 0, 0), -1)
        cv2.putText(overlay, label, (35, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA)

        if out_path:
            ensure_parent_dir(out_path)
            cv2.imwrite(out_path, overlay)
        return overlay

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        debug_out_path = image_path.replace(".jpg", "_DEBUG.jpg")
        return self.process_frame(img, debug_out_path=debug_out_path, source_name=image_path)


def analyze_state(game_state, top=10):
    pathfinder = MovePathfinder()
    board = game_state.get("board", [])
    targets = game_state.get("gameState", {}).get("targets", {})
    moves_left = pathfinder._parse_moves_left(game_state.get("gameState", {}).get("movesLeft"))
    paths = pathfinder.find_paths(board)
    moves = pathfinder.score_paths(board, targets, paths, moves_left=moves_left)[:top]
    enriched = dict(game_state)
    enriched["analysis"] = {
        "validChains": len(paths),
        "movesLeftParsed": moves_left,
        "bestMoves": [move.to_dict() for move in moves],
    }
    enriched["bestMove"] = moves[0].to_dict() if moves else None
    return enriched


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path)) if path else ""
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_json_result(result_json, out_path):
    ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=4, ensure_ascii=False)


def show_overlay_window(image, title="AIponchik move overlay", wait_ms=0):
    """Show the calculated move in a foreground OpenCV window.

    ADB itself has no drawable phone window, so this is a separate topmost
    helper window that can be placed over/near a scrcpy mirror.
    """
    if image is None:
        return
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    h, w = image.shape[:2]
    max_h = 1200
    scale = min(1.0, max_h / float(h)) if h else 1.0
    cv2.resizeWindow(title, max(1, int(w * scale)), max(1, int(h * scale)))
    try:
        cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass
    cv2.imshow(title, image)
    if wait_ms == 0:
        print("Overlay window is open. Press any key in that window, or close the window, to continue.")
        while True:
            key = cv2.waitKey(100)
            if key != -1:
                break
            try:
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    return
            except cv2.error:
                return
    else:
        cv2.waitKey(wait_ms)
    cv2.destroyWindow(title)


class MoveAssistantGui:
    """Persistent OpenCV UI for live screen watching and move suggestions."""

    def __init__(self, title="AIponchik live assistant", enabled=True):
        self.title = title
        self.enabled = enabled
        if enabled:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)

    def render(self, frame, status, result=None, diff=None, stable_count=0):
        if not self.enabled or frame is None:
            return True

        canvas = frame.copy()
        h, w = canvas.shape[:2]
        panel_h = max(170, int(h * 0.13))
        cv2.rectangle(canvas, (0, 0), (w, panel_h), (20, 20, 20), -1)
        cv2.putText(canvas, status, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 255, 255), 3, cv2.LINE_AA)
        details = [f"stable frames: {stable_count}"]
        if diff is not None:
            details.append(f"screen diff: {diff:.2f}")
        if result:
            game_state = result.get("gameState", {})
            best = result.get("bestMove")
            details.append(f"moves left: {game_state.get('movesLeft', '?')}")
            if best:
                details.append(f"best: {best.get('item', '?')} x{best.get('length', '?')} score={best.get('score', '?')}")
            else:
                details.append("best: no valid chain")
        details.append("q/Esc: quit")
        for index, line in enumerate(details):
            cv2.putText(canvas, line, (24, 88 + index * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        max_h = 1200
        scale = min(1.0, max_h / float(h)) if h else 1.0
        cv2.resizeWindow(self.title, max(1, int(w * scale)), max(1, int(h * scale)))
        try:
            cv2.setWindowProperty(self.title, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass
        cv2.imshow(self.title, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            return False
        try:
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def close(self):
        if self.enabled:
            try:
                cv2.destroyWindow(self.title)
            except cv2.error:
                pass


def parse_args():
    cli = argparse.ArgumentParser(description="Parse Cookie Cats board screenshots and suggest the best chain.")
    cli.add_argument("--image", help="Analyze a single local screenshot.")
    cli.add_argument("--dir", default="test_images", help="Directory with .jpg screenshots for batch mode.")
    cli.add_argument("--adb", action="store_true", help="Capture the current Android screen through ADB.")
    cli.add_argument("--stream", help="Read one stable frame from a scrcpy/OpenCV stream, e.g. /dev/video2.")
    cli.add_argument("--adb-path", default="adb", help="Path to adb executable.")
    cli.add_argument("--serial", help="ADB device serial if multiple devices are connected.")
    cli.add_argument("--stable-frames", type=int, default=3, help="Stable frame count required before analysis.")
    cli.add_argument("--stable-threshold", type=float, default=1.5, help="Mean pixel diff threshold for stable screen detection.")
    cli.add_argument("--timeout", type=float, default=20.0, help="Seconds to wait for a stable ADB screen.")
    cli.add_argument("--out", help="Output JSON path for --image, --adb, or --stream mode.")
    cli.add_argument("--overlay", help="Output image path with the best calculated move drawn over the screen.")
    cli.add_argument("--show-overlay-window", action="store_true",
                     help="Show the calculated move in a topmost OpenCV window after saving it.")
    cli.add_argument("--gui", action="store_true",
                     help="Run a persistent GUI that waits for stable screens and updates the suggested move.")
    cli.add_argument("--watch", action="store_true",
                     help="Keep watching ADB/scrcpy after each stable analysis instead of exiting after one frame.")
    cli.add_argument("--window-ms", type=int, default=0,
                     help="Milliseconds to keep --show-overlay-window open; 0 waits for a key press.")
    cli.add_argument("--top", type=int, default=10, help="Number of suggested moves to include.")
    cli.add_argument("--play-move", action="store_true",
                     help="Experimental: after ADB analysis, draw the best move on the phone with input swipe segments.")
    cli.add_argument("--swipe-duration-ms", type=int, default=120,
                     help="Duration of each experimental ADB swipe segment for --play-move.")
    cli.add_argument("--simulate", action="store_true",
                     help="Run deterministic random full-game simulations instead of parsing screenshots.")
    cli.add_argument("--simulate-games", type=int, default=50, help="Number of simulated games to run.")
    cli.add_argument("--simulate-seed", type=int, default=20260601, help="Random seed for simulations.")
    cli.add_argument("--etalon-dir", default="etalon_images", help="Directory with ideal etalon JSON files for simulations.")
    return cli.parse_args()


def analyze_image_file(parser, image_path, out_path=None, top=10, overlay_path=None, show_window=False, window_ms=0):
    print(f"Анализ: {image_path} ...", end=" ")
    frame = cv2.imread(image_path)
    debug_out_path = image_path.replace(".jpg", "_DEBUG.jpg")
    result_json = parser.process_frame(frame, debug_out_path=debug_out_path, source_name=image_path)
    if not result_json:
        print("Ошибка")
        return None
    result_json = analyze_state(result_json, top=top)
    out_path = out_path or image_path.replace(".jpg", ".json")
    write_json_result(result_json, out_path)
    overlay_img = None
    if result_json.get("bestMove"):
        if overlay_path:
            overlay_img = parser.draw_move_overlay(frame, result_json["bestMove"], overlay_path)
        elif show_window:
            overlay_img = parser.draw_move_overlay(frame, result_json["bestMove"])
    if show_window and overlay_img is not None:
        show_overlay_window(overlay_img, wait_ms=window_ms)
    best = result_json.get("bestMove")
    best_text = f" лучший ход: {best['item']} x{best['length']} score={best['score']}" if best else " ходов не найдено"
    overlay_text = f"; overlay={overlay_path}" if overlay_path else ""
    print(f"Готово! Сохранено в {out_path};{best_text}{overlay_text}")
    return result_json


def analyze_captured_frame(parser, frame, out_path, debug_label, args):
    debug_path = out_path.rsplit(".", 1)[0] + "_DEBUG.jpg"
    result_json = parser.process_frame(frame, debug_out_path=debug_path, source_name=debug_label)
    if not result_json:
        print(f"{debug_label}: стабильный кадр получен, но поле не найдено; продолжаю ожидание.")
        return None
    result_json = analyze_state(result_json, top=args.top)
    write_json_result(result_json, out_path)
    overlay_path = args.overlay or out_path.rsplit(".", 1)[0] + "_MOVE.jpg"
    overlay_img = None
    if result_json.get("bestMove"):
        overlay_img = parser.draw_move_overlay(frame, result_json["bestMove"], overlay_path)
    print(f"Готово! Сохранено в {out_path}; debug={debug_path}; overlay={overlay_path}")
    if args.show_overlay_window and overlay_img is not None:
        show_overlay_window(overlay_img, wait_ms=args.window_ms)
    if result_json.get("bestMove"):
        print(json.dumps(result_json["bestMove"], ensure_ascii=False, indent=2))
    return result_json


def run_live_assistant(parser, reader, args, debug_label, default_out):
    """Continuously wait for stable frames, analyze them, and keep a GUI updated."""
    out_path = args.out or default_out
    gui = MoveAssistantGui(enabled=args.gui)
    last_signature = None
    last_result = None
    last_display_frame = None
    last_board_signature = None

    def on_waiting_frame(frame, diff, stable_count):
        return gui.render(
            frame,
            "Ожидание стабилизации экрана...",
            diff=diff,
            stable_count=stable_count,
        )

    try:
        while True:
            try:
                frame = reader.wait_for_stable_frame(
                    stable_frames=args.stable_frames,
                    threshold=args.stable_threshold,
                    timeout=args.timeout,
                    status_callback=on_waiting_frame if args.gui else None,
                )
            except TimeoutError as exc:
                print(f"{debug_label}: {exc}; продолжаю ожидание.")
                continue

            signature = cv2.resize(frame, (64, 128), interpolation=cv2.INTER_AREA).tobytes()
            if signature == last_signature and last_display_frame is not None:
                if args.gui and not gui.render(
                    last_display_frame,
                    "Экран стабилен, изменений нет; показываю последний рассчитанный ход...",
                    result=last_result,
                    stable_count=args.stable_frames,
                ):
                    break
                if not args.watch:
                    break
                time.sleep(0.5)
                continue
            last_signature = signature

            result = analyze_captured_frame(parser, frame, out_path, debug_label, args)
            display_frame = frame
            status = "Поле не найдено на стабильном экране; жду следующее состояние..."
            if result:
                board_signature = json.dumps(result.get("board", []), ensure_ascii=False, sort_keys=True)
                if board_signature == last_board_signature and last_display_frame is not None:
                    display_frame = last_display_frame
                    status = "Поле не изменилось; не пересчитываю ход и оставляю прошлую подсказку"
                    result = last_result
                else:
                    last_board_signature = board_signature
                    if result.get("bestMove"):
                        display_frame = parser.draw_move_overlay(frame, result["bestMove"], args.overlay)
                        status = "Стабильный экран: лучший ход рассчитан"
                        if args.play_move and isinstance(reader, AdbScreenReader):
                            segments = reader.play_move(parser, result["bestMove"], duration_ms=args.swipe_duration_ms)
                            print(f"ADB: experimental move sent as {segments} swipe segment(s).")
                    else:
                        status = "Стабильный экран: ходов не найдено"
                    last_result = result
                    last_display_frame = display_frame
            if args.gui and not gui.render(display_frame, status, result=result, stable_count=args.stable_frames):
                break
            if not args.watch and not args.gui:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Остановлено пользователем.")
    finally:
        gui.close()


def run_simulation(args):
    config = SimulationConfig(games=args.simulate_games, seed=args.simulate_seed)
    simulator = RandomGameSimulator(etalon_dir=args.etalon_dir, config=config)
    report = simulator.run_many(args.simulate_games)
    out_path = args.out or os.path.join("run_outputs", "simulation_report.json")
    write_json_result(report, out_path)
    summary = report["summary"]
    print(
        f"Simulation complete: {summary['wins']}/{summary['games']} wins "
        f"({summary['successRate']}%). Report: {out_path}"
    )
    return report


def analyze_adb_screen(parser, args):
    print("ADB: подключаюсь к устройству и жду стабильности экрана...")
    reader = AdbScreenReader(adb_path=args.adb_path, serial=args.serial)
    if args.watch or args.gui:
        return run_live_assistant(parser, reader, args, "adb", os.path.join("test_images", "adb_capture.json"))
    frame = reader.wait_for_stable_frame(
        stable_frames=args.stable_frames,
        threshold=args.stable_threshold,
        timeout=args.timeout,
    )
    out_path = args.out or os.path.join("test_images", "adb_capture.json")
    result = analyze_captured_frame(parser, frame, out_path, "adb", args)
    if args.play_move and result and result.get("bestMove"):
        segments = reader.play_move(parser, result["bestMove"], duration_ms=args.swipe_duration_ms)
        print(f"ADB: experimental move sent as {segments} swipe segment(s).")
    return result


def analyze_scrcpy_stream(parser, args):
    print(f"scrcpy/OpenCV: читаю поток {args.stream} и жду стабильности экрана...")
    reader = ScrcpyStreamReader(args.stream)
    if args.watch or args.gui:
        return run_live_assistant(parser, reader, args, "scrcpy", os.path.join("test_images", "scrcpy_capture.json"))
    frame = reader.wait_for_stable_frame(
        stable_frames=args.stable_frames,
        threshold=args.stable_threshold,
        timeout=args.timeout,
    )
    out_path = args.out or os.path.join("test_images", "scrcpy_capture.json")
    return analyze_captured_frame(parser, frame, out_path, "scrcpy", args)


if __name__ == "__main__":
    args = parse_args()
    parser = GameBoardParser()

    if args.simulate:
        run_simulation(args)
    elif args.adb:
        analyze_adb_screen(parser, args)
    elif args.stream:
        analyze_scrcpy_stream(parser, args)
    elif args.image:
        analyze_image_file(
            parser,
            args.image,
            out_path=args.out,
            top=args.top,
            overlay_path=args.overlay,
            show_window=args.show_overlay_window,
            window_ms=args.window_ms,
        )
    else:
        images = glob.glob(os.path.join(args.dir, "*.jpg"))
        images = [img_path for img_path in images if "_DEBUG.jpg" not in img_path]
        if not images:
            print(f"В папке {args.dir} не найдено файлов .jpg!")
        for img_path in images:
            analyze_image_file(parser, img_path, top=args.top)
