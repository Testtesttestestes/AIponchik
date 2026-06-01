"""Train the experimental neural move-ranking policy.

The trained model is saved as JSON and is not loaded by ``game_parser.py`` yet.
This lets us iterate on training quality without changing live gameplay.
"""

import argparse
import json

from neural_policy import TrainingConfig, train_policy
from game_parser import ensure_parent_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train AIponchik neural move-ranking policy.")
    parser.add_argument("--etalon-dir", default="etalon_images", help="Directory with ideal parsed JSON states.")
    parser.add_argument("--out", default="models/policy_network.json", help="Output model JSON path.")
    parser.add_argument("--metrics-out", default="run_outputs/policy_training_metrics.json", help="Training metrics JSON path.")
    parser.add_argument("--train-games", type=int, default=60, help="Teacher-played simulated games for training.")
    parser.add_argument("--val-games", type=int, default=20, help="Held-out simulated games for validation.")
    parser.add_argument("--candidates", type=int, default=16, help="Candidate moves per state used for ranking labels.")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum training epochs.")
    parser.add_argument("--hidden-units", type=int, default=48, help="Hidden layer size.")
    parser.add_argument("--learning-rate", type=float, default=0.003, help="Adam learning rate.")
    parser.add_argument("--l2", type=float, default=0.0005, help="L2 regularization strength.")
    parser.add_argument("--dropout", type=float, default=0.08, help="Hidden-layer dropout during training.")
    parser.add_argument("--patience", type=int, default=12, help="Early-stopping patience on validation loss.")
    parser.add_argument("--seed", type=int, default=20260601, help="Deterministic training seed.")
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
    )
    model, metrics = train_policy(config, etalon_dir=args.etalon_dir)
    model.save(args.out, metrics=metrics, config=config)
    ensure_parent_dir(args.metrics_out)
    with open(args.metrics_out, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(
        "Training complete: "
        f"val top-1 agreement={metrics['valTop1Agreement']}%, "
        f"best val loss={metrics['bestValLoss']:.5f}, "
        f"model={args.out}, metrics={args.metrics_out}"
    )


if __name__ == "__main__":
    main()
