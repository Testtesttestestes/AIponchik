"""Fair paired benchmark for the heuristic solver and neural policy blends.

Each tested ranker receives the exact same synthetic game inputs by replaying
one deterministic seed per game.  This avoids comparing two different random
boards that merely share the same top-level simulation seed.
"""

import argparse
import json
from typing import Dict, List, Optional

from game_parser import (
    DEFAULT_POLICY_CANDIDATES,
    DEFAULT_POLICY_MODEL,
    NeuralPolicyMoveSelector,
    RandomGameSimulator,
    SimulationConfig,
    ensure_parent_dir,
)


def parse_blends(raw: str) -> List[float]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = float(chunk)
        if not 0.0 <= value <= 1.0:
            raise argparse.ArgumentTypeError("Blend values must be in [0.0, 1.0]")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("At least one blend value is required")
    return values


def summarize(results):
    by_difficulty: Dict[str, Dict[str, float]] = {}
    total_moves = 0
    direct_target_moves = 0
    off_target_moves = 0
    short_off_target_moves = 0
    guarded_moves = 0

    for result in results:
        bucket = by_difficulty.setdefault(result.difficulty, {"games": 0, "wins": 0})
        bucket["games"] += 1
        bucket["wins"] += int(result.success)

        for turn in result.turns:
            total_moves += 1
            move = turn.get("move") or {}
            collected = turn.get("collected") or {}
            collected_targets = sum(int(value) for value in collected.values())
            if collected_targets > 0:
                direct_target_moves += 1
            else:
                off_target_moves += 1
                if int(move.get("length", 0)) < 4:
                    short_off_target_moves += 1
            if move.get("selectedBy") == "neuralPolicyTargetGuard":
                guarded_moves += 1

    for bucket in by_difficulty.values():
        bucket["successRate"] = round(bucket["wins"] / bucket["games"] * 100.0, 2) if bucket["games"] else 0.0
    wins = sum(1 for result in results if result.success)
    return {
        "games": len(results),
        "wins": wins,
        "losses": len(results) - wins,
        "successRate": round(wins / len(results) * 100.0, 2) if results else 0.0,
        "byDifficulty": by_difficulty,
        "moveDiagnostics": {
            "totalMoves": total_moves,
            "directTargetMoves": direct_target_moves,
            "offTargetMoves": off_target_moves,
            "shortOffTargetMoves": short_off_target_moves,
            "shortOffTargetRate": round(short_off_target_moves / total_moves * 100.0, 2) if total_moves else 0.0,
            "targetGuardedMoves": guarded_moves,
        },
    }


def run_paired_games(games: int, base_seed: int, etalon_dir: str, selector: Optional[NeuralPolicyMoveSelector] = None):
    results = []
    for game_index in range(1, games + 1):
        config = SimulationConfig(games=1, seed=base_seed + game_index)
        simulator = RandomGameSimulator(etalon_dir=etalon_dir, config=config, move_selector=selector)
        results.append(simulator.run_one(game_index))
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare heuristic and neural policy on identical simulated games.")
    parser.add_argument("--games", type=int, default=50, help="Number of paired games to run.")
    parser.add_argument("--base-seed", type=int, default=20260601, help="Base seed; game i uses base-seed + i.")
    parser.add_argument("--etalon-dir", default="etalon_images", help="Directory with ideal etalon JSON files.")
    parser.add_argument("--policy-model", default=DEFAULT_POLICY_MODEL, help="Neural policy JSON path.")
    parser.add_argument("--policy-candidates", type=int, default=DEFAULT_POLICY_CANDIDATES, help="Candidate moves per neural decision.")
    parser.add_argument("--blends", type=parse_blends, default=parse_blends("0.0,0.5"),
                        help="Comma-separated heuristic blend values to test for the neural policy.")
    parser.add_argument("--out", default="run_outputs/policy_benchmark_comparison.json", help="Output JSON report path.")
    args = parser.parse_args()

    algorithm_results = run_paired_games(args.games, args.base_seed, args.etalon_dir)
    algorithm_summary = summarize(algorithm_results)
    policies = []

    for blend in args.blends:
        selector = NeuralPolicyMoveSelector(
            model_path=args.policy_model,
            candidates_per_move=args.policy_candidates,
            blend_heuristic=blend,
        )
        results = run_paired_games(args.games, args.base_seed, args.etalon_dir, selector=selector)
        summary = summarize(results)
        policies.append({
            "name": f"neuralPolicy_blend_{blend:g}",
            "ranker": "neuralPolicy",
            "blendHeuristic": blend,
            "policyModel": args.policy_model,
            "policyCandidateLimit": args.policy_candidates,
            "summary": summary,
            "deltaVsAlgorithm": {
                "wins": summary["wins"] - algorithm_summary["wins"],
                "successRate": round(summary["successRate"] - algorithm_summary["successRate"], 2),
            },
        })

    report = {
        "config": {
            "games": args.games,
            "baseSeed": args.base_seed,
            "perGameSeed": "baseSeed + gameIndex",
            "etalonDir": args.etalon_dir,
            "policyModel": args.policy_model,
            "policyCandidateLimit": args.policy_candidates,
            "blends": args.blends,
        },
        "algorithm": {
            "ranker": "heuristic",
            "summary": algorithm_summary,
        },
        "policies": policies,
    }

    ensure_parent_dir(args.out)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(
        f"Algorithm: {algorithm_summary['wins']}/{algorithm_summary['games']} "
        f"({algorithm_summary['successRate']}%)"
    )
    for policy in policies:
        summary = policy["summary"]
        delta = policy["deltaVsAlgorithm"]
        print(
            f"{policy['name']}: {summary['wins']}/{summary['games']} "
            f"({summary['successRate']}%), delta {delta['wins']:+d} wins / {delta['successRate']:+.2f} pp"
        )
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
