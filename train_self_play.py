"""Reinforcement Learning (Self-Play) training loop for AIponchik."""

import argparse

from neural_policy import TrainingConfig, SelfPlayDatasetBuilder, NeuralMovePolicy, evaluate_policy_games, FEATURE_NAMES

def parse_args():
    parser = argparse.ArgumentParser(description="Train AIponchik neural policy via Self-Play.")
    parser.add_argument("--etalon-dir", default="etalon_images", help="Directory with ideal parsed JSON states.")
    parser.add_argument("--out", default="models/policy_network_rl.json", help="Output model JSON path.")
    parser.add_argument("--generations", type=int, default=5, help="Number of self-play generation loops to run.")
    parser.add_argument("--games-per-gen", type=int, default=280, help="Games to play against itself per generation.")
    parser.add_argument("--eval-games", type=int, default=70, help="Games to evaluate the final model win rate.")
    parser.add_argument("--start-epsilon", type=float, default=0.15, help="Starting random move percentage (exploration).")
    parser.add_argument("--end-epsilon", type=float, default=0.05, help="Ending random move percentage.")
    parser.add_argument("--resume", default=None, help="Path to existing model JSON to continue RL training.")
    
    # Вернули ваши привычные параметры!
    parser.add_argument("--learning-rate", type=float, default=0.001, help="AdamW learning rate.")
    parser.add_argument("--candidates", type=int, default=16, help="Candidate moves per state used for ranking labels.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per generation.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    config = TrainingConfig(
        train_games=args.games_per_gen,
        val_games=max(20, args.games_per_gen // 5),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        candidates_per_state=args.candidates
    )
    
    if args.resume:
        print(f"🔄 Resuming RL training from {args.resume}...")
        model = NeuralMovePolicy.load(args.resume)
        model.model = model.model.to(model.device)
    else:
        print("✨ Initializing fresh RL agent...")
        model = NeuralMovePolicy(len(FEATURE_NAMES), hidden_units=128, seed=config.seed)

    builder = SelfPlayDatasetBuilder(etalon_dir=args.etalon_dir, config=config)

    for gen in range(1, args.generations + 1):
        # Плавно снижаем процент случайных ходов от start до end
        current_epsilon = args.start_epsilon - ((args.start_epsilon - args.end_epsilon) * (gen / args.generations))
        builder.config = TrainingConfig(**{**config.__dict__, "epsilon": current_epsilon})
        
        print(f"\n{'='*40}")
        print(f"🧬 GENERATION {gen}/{args.generations} (Exploration ε: {current_epsilon:.2f})")
        print(f"{'='*40}")
        
        builder.set_model(model)

        print("Generating Training Data (Self-Play)...")
        train_data = builder.build(config.train_games, seed_offset=gen * 100_000)
        
        print("Generating Validation Data (Self-Play)...")
        val_data = builder.build(config.val_games, seed_offset=gen * 200_000)
        
        print(f"Gen {gen} Stats: Won {train_data.successful_games}/{config.train_games} training games. "
              f"Collected {train_data.size} successful state-action pairs.")

        if train_data.size > 0:
            metrics = model.fit(train_data, val_data, builder.config, save_path=args.out)
            win_rate = evaluate_policy_games(model, games=10, etalon_dir=args.etalon_dir, candidates_per_state=args.candidates)
            print(f"Gen {gen} Quick Eval Win Rate: {win_rate}% | Best Val Loss: {metrics['bestValLoss']:.4f}")
        else:
            print("⚠️ Agent won 0 games this generation. Cannot update weights.")

    print("\nStarting final rigorous post-training evaluation...")
    final_win_rate = evaluate_policy_games(model, games=args.eval_games, etalon_dir=args.etalon_dir, candidates_per_state=args.candidates)
    print(f"🏁 Final Self-Play Agent Win Rate: {final_win_rate}% over {args.eval_games} games.")
    print(f"Model saved to {args.out}")

if __name__ == "__main__":
    main()