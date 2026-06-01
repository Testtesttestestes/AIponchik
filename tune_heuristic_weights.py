"""Bayesian optimization for AIponchik heuristic weights.

The script uses Optuna when available and falls back to deterministic random
search otherwise, so it can still run in constrained environments.  Each trial
plays full simulated games and maximizes held-out win rate.
"""

import argparse
import json
import random
from dataclasses import asdict

try:
    import optuna
except ImportError:  # pragma: no cover - exercised only without optional dependency
    optuna = None

from game_parser import HeuristicWeights, MovePathfinder, RandomGameSimulator, SimulationConfig, write_json_result


def build_pathfinder(weights, args):
    return MovePathfinder(
        max_paths_per_item=args.max_paths_per_item,
        weights=weights,
        search_depth=args.search_depth,
        rollout_samples=args.rollout_samples,
    )


def evaluate_weights(weights, args, seed):
    simulator = RandomGameSimulator(
        etalon_dir=args.etalon_dir,
        config=SimulationConfig(games=args.games_per_trial, seed=seed),
        pathfinder=build_pathfinder(weights, args),
    )
    report = simulator.run_many(args.games_per_trial)
    return report["summary"]["successRate"], report


def suggest_weights(trial=None, rng=None):
    if trial is not None:
        return HeuristicWeights(
            immediate=trial.suggest_float("immediate", 0.6, 1.8),
            future=trial.suggest_float("future", 0.0, 0.8),
            cluster=trial.suggest_float("cluster", 0.5, 8.0),
            orphan=trial.suggest_float("orphan", 0.0, 12.0),
            ice_target=trial.suggest_float("ice_target", 1.0, 20.0),
            stochastic=trial.suggest_float("stochastic", 0.0, 1.2),
            lookahead=trial.suggest_float("lookahead", 0.0, 1.2),
        )
    return HeuristicWeights(
        immediate=rng.uniform(0.6, 1.8),
        future=rng.uniform(0.0, 0.8),
        cluster=rng.uniform(0.5, 8.0),
        orphan=rng.uniform(0.0, 12.0),
        ice_target=rng.uniform(1.0, 20.0),
        stochastic=rng.uniform(0.0, 1.2),
        lookahead=rng.uniform(0.0, 1.2),
    )


def run_optuna(args):
    best_report = {}

    def objective(trial):
        weights = suggest_weights(trial=trial)
        rate, report = evaluate_weights(weights, args, args.seed + trial.number)
        trial.set_user_attr("weights", asdict(weights))
        trial.set_user_attr("wins", report["summary"]["wins"])
        best_report[trial.number] = report["summary"]
        return rate

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials)
    return {
        "optimizer": "optuna-tpe",
        "bestValue": study.best_value,
        "bestWeights": study.best_trial.user_attrs["weights"],
        "bestTrial": study.best_trial.number,
        "trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "weights": trial.user_attrs.get("weights"),
                "wins": trial.user_attrs.get("wins"),
            }
            for trial in study.trials
        ],
    }


def run_random_search(args):
    rng = random.Random(args.seed)
    trials = []
    best = None
    for number in range(args.trials):
        weights = suggest_weights(rng=rng)
        rate, report = evaluate_weights(weights, args, args.seed + number)
        row = {"number": number, "value": rate, "weights": asdict(weights), "wins": report["summary"]["wins"]}
        trials.append(row)
        if best is None or rate > best["value"]:
            best = row
    return {
        "optimizer": "random-search-fallback",
        "bestValue": best["value"],
        "bestWeights": best["weights"],
        "bestTrial": best["number"],
        "trials": trials,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Tune AIponchik heuristic weights with Optuna Bayesian optimization.")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--games-per-trial", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--etalon-dir", default="etalon_images")
    parser.add_argument("--search-depth", type=int, default=2)
    parser.add_argument("--rollout-samples", type=int, default=0)
    parser.add_argument("--max-paths-per-item", type=int, default=15)
    parser.add_argument("--out", default="run_outputs/heuristic_tuning.json")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_optuna(args) if optuna is not None else run_random_search(args)
    result["config"] = vars(args)
    write_json_result(result, args.out)
    print(json.dumps({"bestValue": result["bestValue"], "bestWeights": result["bestWeights"], "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
