"""Train the experimental neural move-ranking policy.

The trained model is saved as JSON and is not loaded by ``game_parser.py`` yet.
This lets us iterate on training quality without changing live gameplay.
"""

import argparse
import json

from neural_policy_randomforest import TrainingConfig, evaluate_policy_games, train_policy
from game_parser import ensure_parent_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train AIponchik neural move-ranking policy.")
    parser.add_argument("--etalon-dir", default="etalon_images", help="Directory with ideal parsed JSON states.")
    parser.add_argument("--out", default="models/policy_network.json", help="Output model JSON path.")
    parser.add_argument("--metrics-out", default="run_outputs/policy_training_metrics.json", help="Training metrics JSON path.")
    parser.add_argument("--train-games", type=int, default=210, help="Teacher-played simulated games attempted for training; losing games are discarded by default.")
    parser.add_argument("--val-games", type=int, default=70, help="Held-out simulated games attempted for validation; losing games are discarded by default.")
    parser.add_argument("--candidates", type=int, default=16, help="Candidate moves per state used for ranking labels.")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum training epochs.")
    parser.add_argument("--hidden-units", type=int, default=128, help="Hidden layer size (increased for PyTorch).")
    parser.add_argument("--learning-rate", type=float, default=0.003, help="AdamW learning rate.")
    parser.add_argument("--l2", type=float, default=0.0005, help="Weight decay strength.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Hidden-layer dropout during training.")
    parser.add_argument("--patience", type=int, default=15, help="Early-stopping patience on validation loss.")
    parser.add_argument("--seed", type=int, default=20260601, help="Deterministic training seed.")
    parser.add_argument("--eval-games", type=int, default=50, help="Simulated games to evaluate with the trained policy after training.")
    parser.add_argument("--eval-seed", type=int, default=20260602, help="Held-out seed for post-training win-rate evaluation.")
    parser.add_argument("--target-win-rate", type=float, default=60.0, help="Target simulated win rate percentage to report against.")
    parser.add_argument("--blend-heuristic", type=float, default=0.0, help="Optional heuristic blend for evaluation only; 0 uses the neural policy alone.")
    parser.add_argument("--include-losing-games", action="store_true",
                        help="Keep failed teacher games in the training dataset. By default only successful games are used.")
    parser.add_argument("--resume", default=None, help="Path to existing model JSON to resume training.")
    
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig(
        train_games=args.train_games,
        val_games=args.val_games,
        candidates_per_state=args.candidates,
        epochs=args.epochs,
        hidden_units=args.hidden_units,
        learning_rate=args.learning_rate,
        l2=args.l2,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
        successful_games_only=not args.include_losing_games,
    )
    
    model, metrics = train_policy(
        config, 
        etalon_dir=args.etalon_dir, 
        save_path=args.out,
        resume_path=args.resume 
    )
    
    print("\nStarting post-training evaluation...")
    evaluation = evaluate_policy_games(
        model,
        games=args.eval_games,
        seed=args.eval_seed,
        etalon_dir=args.etalon_dir,
        candidates_per_state=args.candidates,
        blend_heuristic=args.blend_heuristic,
    )
    metrics["policyEvaluation"] = evaluation
    metrics["targetWinRate"] = args.target_win_rate
    metrics["targetReached"] = evaluation["successRate"] >= args.target_win_rate
    
    model.save(args.out, metrics=metrics, config=config)
    ensure_parent_dir(args.metrics_out)
    
    with open(args.metrics_out, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        
    target_text = "reached" if metrics["targetReached"] else "not reached"
    print(
        f"\nTraining complete: "
        f"val top-1 agreement={metrics['valTop1Agreement']}%, "
        f"policy wins={evaluation['wins']}/{evaluation['games']} ({evaluation['successRate']}%; target {target_text}), "
        f"best val loss={metrics['bestValLoss']:.5f}, "
        f"model={args.out}, metrics={args.metrics_out}"
    )


if __name__ == "__main__":
    main()
