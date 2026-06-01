"""Audit whether the neural policy sees and follows level targets.

This diagnostic intentionally does not change gameplay.  It replays deterministic
heuristic-teacher states, ranks the same candidates with the neural model, and
reports whether the neural top pick collects current targets or drifts to an
off-target item.
"""

import argparse
import collections
import glob
import json
from typing import Dict, List

import numpy as np

from game_parser import (
    DEFAULT_POLICY_CANDIDATES,
    DEFAULT_POLICY_MODEL,
    MovePathfinder,
    NumpyNeuralMovePolicy,
    RandomGameSimulator,
    RuntimeMoveFeatureEncoder,
    SimulationConfig,
    ensure_parent_dir,
)
from neural_policy import FEATURE_NAMES

TARGET_FEATURES = {
    "moves_left_norm",
    "move_item_target_fraction",
    "move_ice_target_fraction",
    "own_target_urgency",
    "ice_target_urgency",
    "max_other_target_urgency",
    *(f"target_remaining_{target}" for target in RuntimeMoveFeatureEncoder.TARGETS),
}


def load_etalon_states(etalon_dir: str) -> List[Dict]:
    states = []
    for path in sorted(glob.glob(f"{etalon_dir}/*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        state["_source"] = path
        states.append(state)
    if not states:
        raise RuntimeError(f"No etalon JSON files found in {etalon_dir}")
    return states


def active_target_names(pathfinder: MovePathfinder, targets: Dict) -> List[str]:
    normalized = pathfinder._normalize_targets(targets)
    return sorted(name for name, data in normalized.items() if data["remaining"] > 0)


def direct_target_yield(pathfinder: MovePathfinder, state: Dict, move) -> int:
    board = state.get("board", [])
    targets = pathfinder._normalize_targets(state.get("gameState", {}).get("targets", {}))
    direct = 0

    target = targets.get(move.item)
    if target and target["remaining"] > 0:
        direct += min(move.length, target["remaining"])

    ice_target = targets.get("ice")
    if ice_target and ice_target["remaining"] > 0:
        ice_hits = sum(1 for row, col in move.path if MovePathfinder.has_ice(board[row][col]))
        direct += min(ice_hits, ice_target["remaining"])

    return direct


def summarize_model_features(model_path: str) -> Dict:
    with open(model_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    names = payload.get("featureNames", [])
    std_by_name = dict(zip(names, payload.get("standardization", {}).get("std", [])))
    present_target_features = [name for name in names if name in TARGET_FEATURES]
    missing_target_features = sorted(TARGET_FEATURES - set(names))
    non_constant_target_features = [
        name for name in present_target_features
        if float(std_by_name.get(name, 0.0)) > 1e-4
    ]

    return {
        "modelPath": model_path,
        "featureCount": len(names),
        "featureListMatchesCode": names == list(FEATURE_NAMES),
        "targetAndMoveBudgetFeaturesPresent": present_target_features,
        "targetAndMoveBudgetFeaturesMissing": missing_target_features,
        "nonConstantTargetAndMoveBudgetFeatures": non_constant_target_features,
        "targetFeatureStd": {name: round(float(std_by_name.get(name, 0.0)), 6) for name in present_target_features},
        "metrics": payload.get("metrics", {}),
        "trainingConfig": payload.get("trainingConfig"),
    }


def summarize_etalon_targets(etalon_dir: str) -> Dict:
    pathfinder = MovePathfinder()
    states = load_etalon_states(etalon_dir)
    target_state_counts = collections.Counter()
    level_counts = collections.Counter()
    for state in states:
        level_counts[str(state.get("gameState", {}).get("level", "?"))] += 1
        for name in active_target_names(pathfinder, state.get("gameState", {}).get("targets", {})):
            target_state_counts[name] += 1
    return {
        "states": len(states),
        "levels": dict(sorted(level_counts.items())),
        "activeTargetStateCounts": dict(sorted(target_state_counts.items())),
    }


def _rank_neural(model, encoder, state: Dict, candidates: List) -> tuple:
    features = np.vstack([encoder.encode(state, move) for move in candidates]).astype(np.float32)
    scores = model.predict(features)
    best_index = int(np.argmax(scores))
    return candidates[best_index], float(scores[best_index]), scores


def audit_teacher_rollout_states(
    model_path: str,
    etalon_dir: str,
    games: int,
    seed: int,
    policy_candidates: int,
    sample_limit: int,
) -> Dict:
    model = NumpyNeuralMovePolicy.load(model_path)
    encoder = RuntimeMoveFeatureEncoder()
    pathfinder = MovePathfinder(max_paths_per_item=20, rollout_samples=0)
    simulator = RandomGameSimulator(
        etalon_dir=etalon_dir,
        config=SimulationConfig(seed=seed),
        pathfinder=pathfinder,
    )

    totals = collections.Counter()
    teacher_by_item = collections.Counter()
    neural_by_item = collections.Counter()
    neural_item_when_not_target = collections.Counter()
    teacher_item_when_not_target = collections.Counter()
    target_presence = collections.defaultdict(collections.Counter)
    samples = []

    for game_index in range(1, games + 1):
        template = simulator.etalon_states[int(simulator.rng.integers(0, len(simulator.etalon_states)))]
        difficulty = list(simulator.DIFFICULTIES)[(game_index - 1) % len(simulator.DIFFICULTIES)]
        modifiers = simulator.DIFFICULTIES[difficulty]
        level = str(template.get("gameState", {}).get("level", "0"))
        moves_limit = simulator._moves_limit(template, modifiers)
        targets_remaining = simulator._scaled_targets(template, modifiers)
        board, ice_hp = simulator._random_board_from_template(template, modifiers)

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
            )[:policy_candidates]
            if not candidates:
                board, ice_hp = simulator._force_reseed_playable_area(board, ice_hp)
                continue

            teacher_move = candidates[0]
            neural_move, neural_score, scores = _rank_neural(model, encoder, state, candidates)
            teacher_yield = direct_target_yield(pathfinder, state, teacher_move)
            neural_yield = direct_target_yield(pathfinder, state, neural_move)
            active_targets = active_target_names(pathfinder, state["gameState"]["targets"])
            target_candidate_count = sum(1 for move in candidates if direct_target_yield(pathfinder, state, move) > 0)

            totals["states"] += 1
            totals["top1Agreement"] += int(neural_move.path == teacher_move.path)
            totals["teacherTargetMoves"] += int(teacher_yield > 0)
            totals["neuralTargetMoves"] += int(neural_yield > 0)
            totals["statesWithTargetCandidates"] += int(target_candidate_count > 0)
            totals["neuralOffTargetWithTargetCandidates"] += int(neural_yield == 0 and target_candidate_count > 0)
            totals["neuralShortOffTargetWithTargetCandidates"] += int(
                neural_yield == 0 and neural_move.length < 4 and target_candidate_count > 0
            )
            totals["teacherOffTargetWithTargetCandidates"] += int(teacher_yield == 0 and target_candidate_count > 0)
            teacher_by_item[teacher_move.item] += 1
            neural_by_item[neural_move.item] += 1

            if teacher_yield == 0:
                teacher_item_when_not_target[teacher_move.item] += 1
            if neural_yield == 0:
                neural_item_when_not_target[neural_move.item] += 1

            for target in active_targets:
                target_presence[target]["states"] += 1
                target_presence[target]["teacherPickedTarget"] += int(teacher_move.item == target or (target == "ice" and teacher_yield > 0))
                target_presence[target]["neuralPickedTarget"] += int(neural_move.item == target or (target == "ice" and neural_yield > 0))

            if neural_yield == 0 and target_candidate_count > 0 and len(samples) < sample_limit:
                samples.append({
                    "gameIndex": game_index,
                    "turn": turn,
                    "level": level,
                    "difficulty": difficulty,
                    "movesLeft": moves_limit - turn + 1,
                    "activeTargets": active_targets,
                    "targetCandidateCount": target_candidate_count,
                    "teacher": {
                        "item": teacher_move.item,
                        "length": teacher_move.length,
                        "heuristicScore": round(float(teacher_move.score), 3),
                        "directTargetYield": teacher_yield,
                    },
                    "neural": {
                        "item": neural_move.item,
                        "length": neural_move.length,
                        "policyScore": round(neural_score, 6),
                        "heuristicRank": candidates.index(neural_move) + 1,
                        "heuristicScore": round(float(neural_move.score), 3),
                        "directTargetYield": neural_yield,
                    },
                    "bestTargetCandidate": next(
                        (
                            {
                                "item": move.item,
                                "length": move.length,
                                "heuristicRank": index + 1,
                                "heuristicScore": round(float(move.score), 3),
                                "directTargetYield": direct_target_yield(pathfinder, state, move),
                                "policyScore": round(float(scores[index]), 6),
                            }
                            for index, move in enumerate(candidates)
                            if direct_target_yield(pathfinder, state, move) > 0
                        ),
                        None,
                    ),
                })

            # Keep the audited state stream aligned with the training teacher.
            simulator._apply_move(board, ice_hp, teacher_move.path, targets_remaining)
            simulator._refill_board(board, ice_hp)

    states = totals["states"]
    target_presence_summary = {
        name: {
            "states": counter["states"],
            "teacherPickedTargetRate": round(counter["teacherPickedTarget"] / counter["states"] * 100.0, 2),
            "neuralPickedTargetRate": round(counter["neuralPickedTarget"] / counter["states"] * 100.0, 2),
        }
        for name, counter in sorted(target_presence.items())
        if counter["states"]
    }
    return {
        "games": games,
        "seed": seed,
        "policyCandidates": policy_candidates,
        "states": states,
        "top1AgreementRate": round(totals["top1Agreement"] / states * 100.0, 2) if states else 0.0,
        "teacherTargetMoveRate": round(totals["teacherTargetMoves"] / states * 100.0, 2) if states else 0.0,
        "neuralTargetMoveRate": round(totals["neuralTargetMoves"] / states * 100.0, 2) if states else 0.0,
        "statesWithTargetCandidates": totals["statesWithTargetCandidates"],
        "neuralOffTargetWithTargetCandidateRate": round(
            totals["neuralOffTargetWithTargetCandidates"] / max(1, totals["statesWithTargetCandidates"]) * 100.0,
            2,
        ),
        "neuralShortOffTargetWithTargetCandidateRate": round(
            totals["neuralShortOffTargetWithTargetCandidates"] / max(1, totals["statesWithTargetCandidates"]) * 100.0,
            2,
        ),
        "teacherOffTargetWithTargetCandidateRate": round(
            totals["teacherOffTargetWithTargetCandidates"] / max(1, totals["statesWithTargetCandidates"]) * 100.0,
            2,
        ),
        "teacherByItem": dict(sorted(teacher_by_item.items())),
        "neuralByItem": dict(sorted(neural_by_item.items())),
        "teacherOffTargetByItem": dict(sorted(teacher_item_when_not_target.items())),
        "neuralOffTargetByItem": dict(sorted(neural_item_when_not_target.items())),
        "targetPresence": target_presence_summary,
        "offTargetSamples": samples,
    }


def analyze_state_file(model_path: str, state_path: str, policy_candidates: int, top: int = 10) -> Dict:
    with open(state_path, "r", encoding="utf-8") as handle:
        state = json.load(handle)

    model = NumpyNeuralMovePolicy.load(model_path)
    encoder = RuntimeMoveFeatureEncoder()
    pathfinder = MovePathfinder(max_paths_per_item=20, rollout_samples=0)
    board = state.get("board", [])
    targets = state.get("gameState", {}).get("targets", {})
    moves_left = pathfinder._parse_moves_left(state.get("gameState", {}).get("movesLeft"))
    paths = pathfinder.find_paths(board)
    candidates = pathfinder.score_paths(board, targets, paths, moves_left=moves_left)[:policy_candidates]
    if not candidates:
        return {
            "statePath": state_path,
            "movesLeftParsed": moves_left,
            "activeTargets": active_target_names(pathfinder, targets),
            "validChains": len(paths),
            "candidates": [],
        }

    features = np.vstack([encoder.encode(state, move) for move in candidates]).astype(np.float32)
    scores = model.predict(features)
    order = sorted(range(len(candidates)), key=lambda index: (float(scores[index]), candidates[index].length), reverse=True)

    return {
        "statePath": state_path,
        "movesLeftParsed": moves_left,
        "rawTargets": targets,
        "activeTargets": active_target_names(pathfinder, targets),
        "validChains": len(paths),
        "policyCandidates": len(candidates),
        "heuristicBest": {
            "item": candidates[0].item,
            "length": candidates[0].length,
            "heuristicScore": round(float(candidates[0].score), 3),
            "directTargetYield": direct_target_yield(pathfinder, state, candidates[0]),
        },
        "neuralTop": [
            {
                "item": candidates[index].item,
                "length": candidates[index].length,
                "policyScore": round(float(scores[index]), 6),
                "heuristicRank": index + 1,
                "heuristicScore": round(float(candidates[index].score), 3),
                "directTargetYield": direct_target_yield(pathfinder, state, candidates[index]),
            }
            for index in order[:top]
        ],
    }


def build_report(args) -> Dict:
    report = {
        "featureAudit": summarize_model_features(args.policy_model),
        "etalonTargetCoverage": summarize_etalon_targets(args.etalon_dir),
        "teacherRolloutAudit": audit_teacher_rollout_states(
            model_path=args.policy_model,
            etalon_dir=args.etalon_dir,
            games=args.games,
            seed=args.seed,
            policy_candidates=args.policy_candidates,
            sample_limit=args.sample_limit,
        ),
    }
    if args.state_json:
        report["singleStateAudit"] = analyze_state_file(
            model_path=args.policy_model,
            state_path=args.state_json,
            policy_candidates=args.policy_candidates,
            top=args.top,
        )
    return report


def main():
    parser = argparse.ArgumentParser(description="Inspect whether the neural policy sees and follows targets.")
    parser.add_argument("--policy-model", default=DEFAULT_POLICY_MODEL, help="Neural policy JSON path.")
    parser.add_argument("--etalon-dir", default="etalon_images", help="Directory with etalon JSON files.")
    parser.add_argument("--games", type=int, default=30, help="Heuristic-teacher games to audit.")
    parser.add_argument("--seed", type=int, default=20260602, help="Deterministic audit seed.")
    parser.add_argument("--policy-candidates", type=int, default=DEFAULT_POLICY_CANDIDATES, help="Candidates ranked by the policy.")
    parser.add_argument("--sample-limit", type=int, default=10, help="Example off-target decisions to keep in the report.")
    parser.add_argument("--state-json", help="Optional parsed game-state JSON to audit directly.")
    parser.add_argument("--top", type=int, default=10, help="Top neural moves to include for --state-json.")
    parser.add_argument("--out", default="run_outputs/policy_behavior_audit.json", help="Output JSON report path.")
    args = parser.parse_args()

    report = build_report(args)
    ensure_parent_dir(args.out)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    feature_audit = report["featureAudit"]
    rollout = report["teacherRolloutAudit"]
    coverage = report["etalonTargetCoverage"]
    print(f"Feature list matches code: {feature_audit['featureListMatchesCode']}")
    print(
        "Target/move-budget features present: "
        f"{len(feature_audit['targetAndMoveBudgetFeaturesPresent'])}; "
        f"non-constant: {len(feature_audit['nonConstantTargetAndMoveBudgetFeatures'])}"
    )
    print(f"Etalon active target state counts: {coverage['activeTargetStateCounts']}")
    print(
        f"Teacher rollout states: {rollout['states']}; top-1 agreement: {rollout['top1AgreementRate']}%; "
        f"teacher target move rate: {rollout['teacherTargetMoveRate']}%; "
        f"neural target move rate: {rollout['neuralTargetMoveRate']}%"
    )
    print(
        "Neural off-target despite target candidate: "
        f"{rollout['neuralOffTargetWithTargetCandidateRate']}%; "
        f"short off-target: {rollout['neuralShortOffTargetWithTargetCandidateRate']}%"
    )
    print(f"Neural by item: {rollout['neuralByItem']}")
    print(f"Neural off-target by item: {rollout['neuralOffTargetByItem']}")
    if "singleStateAudit" in report:
        single = report["singleStateAudit"]
        print(
            f"State audit: movesLeft={single['movesLeftParsed']}, "
            f"activeTargets={single['activeTargets']}, "
            f"heuristicBest={single.get('heuristicBest')}, "
            f"neuralBest={(single.get('neuralTop') or [None])[0]}"
        )
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
