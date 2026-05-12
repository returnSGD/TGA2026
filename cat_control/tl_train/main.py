"""
《猫语心声》 —— RL策略网络训练主入口

阶段二（第5-8周）：PPO训练

用法:
    # 单猫PPO训练（验证基本需求学习）
    python -m rl_train.main --mode single --cat oreo --steps 500000

    # 多猫自对弈训练（涌现社交行为）
    python -m rl_train.main --mode selfplay --steps 2000000

    # 完整流水线（BC→单猫→自对弈）
    python -m rl_train.main --mode full --steps-single 500000 --steps-selfplay 1500000

    # 评估已训练模型
    python -m rl_train.main --mode eval --checkpoint rl_train/checkpoints/ppo_selfplay_best.pt

    # 性格对比评估
    python -m rl_train.main --mode eval-personality --checkpoint path/to/model.pt

    # BC vs RL 对比
    python -m rl_train.main --mode compare --rl-ckpt path/to/rl.pt --bc-ckpt path/to/bc.pt
"""

from __future__ import annotations
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_train.config import TrainConfig, CAT_PERSONALITIES
from rl_train.trainer import PPOTrainer
from rl_train.evaluate import RLEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="《猫语心声》RL策略网络训练 — PPO框架"
    )
    parser.add_argument("--mode", type=str, default="single",
                       choices=["single", "selfplay", "full", "eval",
                               "eval-personality", "compare"],
                       help="训练/评估模式")
    parser.add_argument("--cat", type=str, default="oreo",
                       help="单猫模式的猫咪ID (xiaoxue/oreo/orange)")
    parser.add_argument("--steps", type=int, default=2_000_000,
                       help="总环境步数（默认200万）")
    parser.add_argument("--steps-single", type=int, default=500_000,
                       help="完整流水线中单猫阶段步数")
    parser.add_argument("--steps-selfplay", type=int, default=1_500_000,
                       help="完整流水线中自对弈阶段步数")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="评估用的模型检查点路径")
    parser.add_argument("--rl-ckpt", type=str, default=None,
                       help="compare模式的RL检查点")
    parser.add_argument("--bc-ckpt", type=str, default=None,
                       help="compare模式的BC检查点")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备 (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="学习率")
    parser.add_argument("--rollout-steps", type=int, default=2000,
                       help="每次rollout收集的步数")
    parser.add_argument("--no-bc", action="store_true",
                       help="不使用BC预训练权重")

    args = parser.parse_args()

    cfg = TrainConfig(
        device=args.device,
        seed=args.seed,
        learning_rate=args.lr,
        steps_per_rollout=args.rollout_steps,
    )
    if args.no_bc:
        cfg.bc_checkpoint = None

    # ── 模式: 单猫训练 ──
    if args.mode == "single":
        trainer = PPOTrainer(cfg)
        trainer.train_single_cat(cat_id=args.cat, total_timesteps=args.steps)

    # ── 模式: 自对弈训练 ──
    elif args.mode == "selfplay":
        trainer = PPOTrainer(cfg)
        trainer.train_self_play(
            cat_ids=["xiaoxue", "oreo", "orange"],
            total_timesteps=args.steps,
        )

    # ── 模式: 完整流水线 ──
    elif args.mode == "full":
        trainer = PPOTrainer(cfg)
        trainer.train_full_pipeline(
            single_cat_timesteps=args.steps_single,
            self_play_timesteps=args.steps_selfplay,
        )

    # ── 模式: 评估 ──
    elif args.mode == "eval":
        if not args.checkpoint:
            # 自动查找最新模型
            ckpt_dir = os.path.join(cfg.model_save_dir)
            candidates = []
            for f in os.listdir(ckpt_dir):
                if f.endswith(".pt"):
                    candidates.append(os.path.join(ckpt_dir, f))
            if candidates:
                candidates.sort(key=os.path.getmtime, reverse=True)
                args.checkpoint = candidates[0]
                print(f"自动选择检查点: {args.checkpoint}")
            else:
                print("错误: 未找到检查点，请用 --checkpoint 指定")
                return

        evaluator = RLEvaluator(args.checkpoint, device=args.device)
        result = evaluator.evaluate_single_cat(
            cat_id=args.cat, num_episodes=20
        )
        evaluator.print_report(result)

    # ── 模式: 性格对比评估 ──
    elif args.mode == "eval-personality":
        if not args.checkpoint:
            ckpt_dir = os.path.join(cfg.model_save_dir)
            candidates = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
            if candidates:
                candidates.sort(key=os.path.getmtime, reverse=True)
                args.checkpoint = candidates[0]
            else:
                print("错误: 未找到检查点")
                return

        evaluator = RLEvaluator(args.checkpoint, device=args.device)
        results = evaluator.evaluate_personality_contrast(num_episodes=10)

        print(f"\n{'═' * 50}")
        print(f"  性格对比报告")
        print(f"{'═' * 50}")
        comparison = results["comparison"]
        for cid in ["xiaoxue", "oreo", "orange"]:
            r = results["per_cat"][cid]
            print(f"\n  {cid}:")
            print(f"    平均奖励: {r['mean_reward']:.2f} | "
                  f"信任: {r['mean_final_trust']:.0f} | "
                  f"压力: {r['mean_final_stress']:.0f}")
            print(f"    Top意图: {comparison.get(f'{cid}_top_intents', [])}")

    # ── 模式: BC vs RL 对比 ──
    elif args.mode == "compare":
        bc_ckpt = args.bc_ckpt or cfg._find_bc_checkpoint()
        rl_ckpt = args.rl_ckpt
        if not bc_ckpt or not rl_ckpt:
            print("错误: 需要同时指定 --bc-ckpt 和 --rl-ckpt")
            return

        evaluator = RLEvaluator(rl_ckpt, device=args.device)
        evaluator.compare_bc_vs_rl(bc_ckpt, rl_ckpt, cat_id=args.cat)


if __name__ == "__main__":
    main()
