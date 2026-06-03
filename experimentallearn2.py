"""Experimental RL Orchestrator Phase 2: Action Discrimination & Weighted Replay."""

import argparse
import random
import sys
from collections import deque
import numpy as np

from neural_policy32 import TrainingConfig, SelfPlayDatasetBuilder, NeuralMovePolicy, evaluate_policy_games, FEATURE_NAMES

def spearman_rank_correlation(a, b):
    if len(a) < 2: return 1.0
    rank_a = np.argsort(np.argsort(a))
    rank_b = np.argsort(np.argsort(b))
    std_a, std_b = np.std(rank_a), np.std(rank_b)
    if std_a == 0 or std_b == 0: return 0.0
    return np.cov(rank_a, rank_b)[0, 1] / (std_a * std_b)

def print_tf_style(prefix, current, total, metrics_dict):
    bar_len = 30
    safe_total = max(1, total) 
    filled = int(round(bar_len * current / float(safe_total)))
    bar = '=' * filled + '-' * (bar_len - filled)
    metrics_str = " - ".join([f"{k}: {v}" for k, v in metrics_dict.items()])
    print(f"{prefix} {current}/{total} [{bar}] - {metrics_str}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--etalon-dir", default="etalon_images")
    parser.add_argument("--out", default="models/policy_network_rl_v5_pairwise.json")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--games-per-gen", type=int, default=500)
    parser.add_argument("--eval-games", type=int, default=200) # Увеличено для точности чекпоинтов!
    parser.add_argument("--start-epsilon", type=float, default=0.19)
    parser.add_argument("--end-epsilon", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    return parser.parse_args()

def main():
    args = parse_args()
    config = TrainingConfig(
        train_games=args.games_per_gen, epochs=args.epochs, learning_rate=args.learning_rate, 
        candidates_per_state=args.candidates, gamma=args.gamma
    )
    
    if args.resume:
        print(f"[INIT] Resuming RL training from {args.resume}...")
        model = NeuralMovePolicy.load(args.resume)
        model.model = model.model.to(model.device)
    else:
        model = NeuralMovePolicy(len(FEATURE_NAMES), hidden_units=128, seed=config.seed)

    generator_pool = deque(maxlen=5)
    generator_pool.append(model.to_dict())
    
    # Буфер теперь также хранит альтернативные действия и индекс поколения
    max_steps_estimate = args.games_per_gen * 4 * 35 
    replay_x = deque(maxlen=max_steps_estimate)
    replay_y = deque(maxlen=max_steps_estimate)
    replay_x_alt = deque(maxlen=max_steps_estimate)
    replay_gen = deque(maxlen=max_steps_estimate) # Для Recency Weighting
    
    builder = SelfPlayDatasetBuilder(etalon_dir=args.etalon_dir, config=config)

    print("\n[INSTRUMENTATION] Building FROZEN Benchmark Set (50 games)...")
    builder.set_model(model) 
    benchmark_data = builder.build(50, seed_offset=9999)
    frozen_benchmark_x = benchmark_data.x
    prev_predictions = model.predict(frozen_benchmark_x) if frozen_benchmark_x.size > 0 else np.array([])
    
    best_winrate = 0.0

    print("\n" + "="*80)
    print("🚀 PHASE 2 EXPERIMENT: ACTION DISCRIMINATION & WEIGHTED REPLAY")
    print("="*80)

    for gen in range(1, args.generations + 1):
        eps = args.start_epsilon - ((args.start_epsilon - args.end_epsilon) * (gen / args.generations))
        builder.config = TrainingConfig(**{**config.__dict__, "epsilon": eps})
        
        print(f"\n🧬 GEN {gen}/{args.generations} | Expl_ε: {eps:.2f} | Pool: {len(generator_pool)} | Buffer: {len(replay_x)}")
        
        builder.set_model(NeuralMovePolicy.from_dict(random.choice(generator_pool)))
        
        print("[DATA] Generative Phase (Self-Play)...")
        train_data = builder.build(config.train_games, seed_offset=gen * 100_000)
        
        print_tf_style("      ", config.train_games, config.train_games, {
            "won": f"{train_data.successful_games}",
            "q_entropy": f"{train_data.q_entropy_mean:.3f}",
            "q_margin": f"{train_data.q_margin_mean:.3f}"
        })

        if train_data.size > 0:
            y_raw = np.array(train_data.y, dtype=np.float32)
            y_norm = (y_raw - np.mean(y_raw)) / (np.std(y_raw) + 1e-8) 
            replay_x.extend(train_data.x)
            replay_y.extend(y_norm)
            replay_x_alt.extend(train_data.x_alt)
            replay_gen.extend([gen] * train_data.size)
            
        buffer_data = builder.build(0)
        buffer_data.x = np.array(replay_x, dtype=np.float32)
        buffer_data.y = np.array(replay_y, dtype=np.float32)
        buffer_data.x_alt = np.array(replay_x_alt, dtype=np.float32)
        
        # WEIGHTED REPLAY: Экспоненциальное затухание по давности (tau=3.0)
        ages = gen - np.array(replay_gen, dtype=np.float32)
        weights = np.exp(-ages / 3.0)
        weights = weights / (np.mean(weights) + 1e-8) # Нормализация весов для стабильности LR
        buffer_data.weights = np.array(weights, dtype=np.float32)

        print("[TRAIN] Optimizing Policy (Huber + Pairwise)...")
        metrics = model.fit(buffer_data, builder.config)
        
        print_tf_style("      ", buffer_data.size, buffer_data.size, {
            "loss_val": f"{metrics.get('valueLoss', 0):.4f}",
            "loss_pair": f"{metrics.get('pairLoss', 0):.4f}",
            "grad_norm_mean": f"{metrics.get('mean_grad_norm', 0):.4f}",
        })
        
        if frozen_benchmark_x.size > 0:
            new_predictions = model.predict(frozen_benchmark_x)
            rank_corr = spearman_rank_correlation(prev_predictions, new_predictions)
            q_var = np.var(new_predictions)
            print_tf_style("[DRIFT]", 50, 50, {
                "rank_correlation": f"{rank_corr:.4f}",
                "q_variance": f"{q_var:.4f}"
            })
            prev_predictions = new_predictions

        print("[EVAL] Validation Phase...")
        # Увеличенная валидация для точного захвата чекпоинта
        win_rate = evaluate_policy_games(model, games=args.eval_games, etalon_dir=args.etalon_dir, candidates=args.candidates)
        print_tf_style("      ", args.eval_games, args.eval_games, {"win_rate": f"{win_rate}%"})
        
        if win_rate >= best_winrate:
            best_winrate = win_rate
            model.save(args.out)
            print(f"🔥 [CHECKPOINT] New best policy saved! (Best: {best_winrate}%)")
            
        if gen % 2 == 0:
            generator_pool.append(model.to_dict())
            print(f"🔄 [DIVERSITY] Current policy added to generator pool.")

    print(f"\n🏁 EXPERIMENT COMPLETE. Final Post-Training Eval (300 games)...")
    final_wr = evaluate_policy_games(model, games=300, etalon_dir=args.etalon_dir, candidates=args.candidates)
    print(f"🏆 Final Eval Win Rate: {final_wr}%")

if __name__ == "__main__":
    main()