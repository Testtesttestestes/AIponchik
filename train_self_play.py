"""Reinforcement Learning (Self-Play) training loop for AIponchik."""

import argparse
import json

from neural_policy2 import TrainingConfig, SelfPlayDatasetBuilder, NeuralMovePolicy, evaluate_policy_games, FEATURE_NAMES
from game_parser import ensure_parent_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Train AIponchik neural policy via Self-Play.")
    parser.add_argument("--etalon-dir", default="etalon_images", help="Directory with ideal parsed JSON states.")
    parser.add_argument("--out", default="models/policy_network_rl.json", help="Output model JSON path.")
    parser.add_argument("--generations", type=int, default=5, help="Number of self-play generation loops to run.")
    parser.add_argument("--games-per-gen", type=int, default=300, help="Games to play against itself per generation.")
    parser.add_argument("--eval-games", type=int, default=50, help="Games to evaluate the final model win rate.")
    parser.add_argument("--start-epsilon", type=float, default=0.20, help="Starting random move percentage (exploration).")
    parser.add_argument("--end-epsilon", type=float, default=0.05, help="Ending random move percentage.")
    parser.add_argument("--resume", default=None, help="Path to existing model JSON to continue RL training.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    config = TrainingConfig(
        train_games=args.games_per_gen,
        val_games=max(20, args.games_per_gen // 5),
        epochs=30,  # Lower epochs per generation to prevent overfitting to a single batch of self-play data
        learning_rate=0.001
    )
    
    if args.resume:
        print(f"🔄 Resuming RL training from {args.resume}...")
        model = NeuralMovePolicy.load(args.resume)
        model.model = model.model.to(model.device)
    else:
        print("✨ Initializing fresh RL agent...")
        model = NeuralMovePolicy(len(FEATURE_NAMES), hidden_units=config.hidden_units, seed=config.seed)

    builder = SelfPlayDatasetBuilder(etalon_dir=args.etalon_dir, config=config)

    for gen in range(1, args.generations + 1):
        # Decay epsilon linearly across generations
        current_epsilon = args.start_epsilon - ((args.start_epsilon - args.end_epsilon) * (gen / args.generations))
        builder.config = TrainingConfig(**{**config.__dict__, "epsilon": current_epsilon})
        
        print(f"\n{'='*40}")
        print(f"🧬 GENERATION {gen}/{args.generations} (Exploration ε: {current_epsilon:.2f})")
        print(f"{'='*40}")
        
        # 1. Provide the builder with the current brain
        builder.set_model(model)

        # 2. Agent plays against itself to generate data
        print("Generating Training Data (Self-Play)...")
        train_data = builder.build(config.train_games, seed_offset=gen * 100_000)
        
        print("Generating Validation Data (Self-Play)...")
        val_data = builder.build(config.val_games, seed_offset=gen * 200_000)
        
        print(f"Gen {gen} Stats: Won {train_data.successful_games}/{config.train_games} training games. "
              f"Collected {train_data.size} successful state-action pairs.")

        # 3. Train on the newly acquired knowledge
        if train_data.size > 0:
            metrics = model.fit(train_data, val_data, builder.config, save_path=args.out)
            
            # Temporary evaluation for tracking progress
            win_rate = evaluate_policy_games(model, games=10, etalon_dir=args.etalon_dir)
            print(f"Gen {gen} Quick Eval Win Rate: {win_rate}% | Best Val Loss: {metrics['bestValLoss']:.4f}")
        else:
            print("⚠️ Agent won 0 games this generation. Cannot update weights. Try increasing --games-per-gen or --start-epsilon.")

    print("\nStarting final rigorous post-training evaluation...")
    final_win_rate = evaluate_policy_games(model, games=args.eval_games, etalon_dir=args.etalon_dir)
    print(f"🏁 Final Self-Play Agent Win Rate: {final_win_rate}% over {args.eval_games} games.")
    print(f"Model saved to {args.out}")

if __name__ == "__main__":
    main()