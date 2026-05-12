"""
《猫语心声》 —— RL策略网络评估

评估训练好的RL策略网络：
- 单猫评估：验证基本需求学习
- 多猫评估：验证性格差异化行为
- 训练曲线可视化
- 意图分布分析
"""

from __future__ import annotations
import sys
import os
import time
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_environment.config import INTENT_LIST, CAT_CONFIGS, STATE_DIM
from rl_environment.environment import SandboxEnvironment
from rl_environment.cat_agent import CatAgent
from rl_environment.personality_filter import PersonalityFilter
from rl_train.config import TrainConfig, CAT_PERSONALITIES
from rl_train.ppo import PPO
from rl_train.env_wrapper import SingleCatEnv, MultiCatEnv


class RLEvaluator:
    """RL策略评估器"""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        # 加载PPO模型
        self.cfg = TrainConfig(device=device)
        self.ppo = PPO(
            state_dim=self.cfg.state_dim,
            personality_dim=self.cfg.personality_dim,
            intent_num=self.cfg.intent_num,
            embed_dim=self.cfg.embed_dim,
            seq_len=self.cfg.seq_len,
            device=device,
        )
        self.ppo.load(checkpoint_path)
        self.ppo.policy.eval()

        self.intent_list = INTENT_LIST

    def evaluate_single_cat(self, cat_id: str = "oreo",
                           num_episodes: int = 20,
                           max_steps: int = 2048,
                           render: bool = False) -> Dict:
        """
        评估单只猫的RL策略性能。

        返回: {episode_rewards, intent_distribution, trust_curve, stress_curve, ...}
        """
        env = SingleCatEnv(cat_id=cat_id, max_steps_per_episode=max_steps)
        personality = env.get_personality()

        episode_rewards = []
        intent_counts = Counter()
        trust_curves = []
        stress_curves = []
        state_seqs = []

        for ep in range(num_episodes):
            obs = env.reset()
            state_buffer = []
            for _ in range(self.cfg.seq_len):
                state_buffer.append(np.zeros(self.cfg.state_dim, dtype=np.float32))
            state_buffer = state_buffer[-self.cfg.seq_len:]

            ep_reward = 0.0
            ep_trust = []
            ep_stress = []
            done = False

            while not done:
                state_buffer.append(obs)
                state_buffer = state_buffer[-self.cfg.seq_len:]
                state_seq = np.stack(state_buffer, axis=0)

                action, _, _ = self.ppo.select_action(state_seq, personality, deterministic=True)

                next_obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated
                obs = next_obs

                intent_counts[info["intent"]] += 1
                ep_trust.append(env.cat.state.trust_level)
                ep_stress.append(env.cat.state.stress_level)

            episode_rewards.append(ep_reward)
            trust_curves.append(ep_trust)
            stress_curves.append(ep_stress)
            state_seqs.append(state_buffer)

        return {
            "cat_id": cat_id,
            "num_episodes": num_episodes,
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "episode_rewards": [float(r) for r in episode_rewards],
            "intent_distribution": {k: v for k, v in intent_counts.most_common()},
            "mean_final_trust": float(np.mean([t[-1] for t in trust_curves if t])),
            "mean_final_stress": float(np.mean([s[-1] for s in stress_curves if s])),
            "trust_curves": [[float(x) for x in c] for c in trust_curves],
            "stress_curves": [[float(x) for x in c] for c in stress_curves],
        }

    def evaluate_personality_contrast(self, num_episodes: int = 10) -> Dict:
        """
        评估同一场景下不同性格猫咪的行为差异。

        对比小雪(怯懦)、奥利奥(傲娇)、橘子(贪吃+活泼)在相同环境中的决策。
        """
        results = {}
        for cat_id in ["xiaoxue", "oreo", "orange"]:
            print(f"  评估 {cat_id}...")
            results[cat_id] = self.evaluate_single_cat(
                cat_id=cat_id, num_episodes=num_episodes
            )

        # 对比分析
        comparison = {
            "rewards": {cid: r["mean_reward"] for cid, r in results.items()},
            "final_trust": {cid: r["mean_final_trust"] for cid, r in results.items()},
            "final_stress": {cid: r["mean_final_stress"] for cid, r in results.items()},
        }

        # 分析每个猫的Top-3意图
        for cid, r in results.items():
            top3 = r["intent_distribution"]
            sorted_intents = sorted(top3.items(), key=lambda x: x[1], reverse=True)[:5]
            total = sum(top3.values())
            comparison[f"{cid}_top_intents"] = [
                f"{i}:{c}/{total}" for i, c in sorted_intents
            ]

        return {"per_cat": results, "comparison": comparison}

    def evaluate_multi_cat(self, cat_ids: List[str] = None,
                          num_episodes: int = 10) -> Dict:
        """评估多猫自对弈策略"""
        cat_ids = cat_ids or ["xiaoxue", "oreo", "orange"]
        env = MultiCatEnv(cat_ids=cat_ids, max_steps_per_episode=2048)
        personalities = env.get_personalities()

        episode_rewards = {cid: [] for cid in cat_ids}
        intent_counts = {cid: Counter() for cid in cat_ids}

        for ep in range(num_episodes):
            obs_dict = env.reset()
            state_buffers = {cid: [] for cid in cat_ids}
            for _ in range(self.cfg.seq_len):
                for cid in cat_ids:
                    state_buffers[cid].append(np.zeros(self.cfg.state_dim, dtype=np.float32))
            for cid in cat_ids:
                state_buffers[cid] = state_buffers[cid][-self.cfg.seq_len:]

            ep_rewards = {cid: 0.0 for cid in cat_ids}
            all_done = False

            while not all_done:
                intent_dict = {}
                for cid in cat_ids:
                    state_buffers[cid].append(obs_dict[cid])
                    state_buffers[cid] = state_buffers[cid][-self.cfg.seq_len:]
                    state_seq = np.stack(state_buffers[cid], axis=0)

                    action, _, _ = self.ppo.select_action(
                        state_seq, personalities[cid], deterministic=True
                    )
                    intent_dict[cid] = action
                    intent_counts[cid][INTENT_LIST[action]] += 1

                next_obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(intent_dict)

                for cid in cat_ids:
                    ep_rewards[cid] += reward_dict[cid]

                obs_dict = next_obs_dict
                all_done = env.all_terminated({cid: term_dict[cid] or trunc_dict[cid]
                                              for cid in cat_ids})

            for cid in cat_ids:
                episode_rewards[cid].append(ep_rewards[cid])

        return {
            "episode_rewards": {cid: [float(r) for r in rews]
                              for cid, rews in episode_rewards.items()},
            "mean_rewards": {cid: float(np.mean(rews)) for cid, rews in episode_rewards.items()},
            "intent_distributions": {cid: dict(ic.most_common(10))
                                    for cid, ic in intent_counts.items()},
        }

    def compare_bc_vs_rl(self, bc_checkpoint: str, rl_checkpoint: str,
                        cat_id: str = "oreo", num_episodes: int = 20) -> Dict:
        """对比BC预训练和RL训练后的策略差异"""
        print(f"\n{'═' * 50}")
        print(f"  BC vs RL 策略对比: {cat_id}")
        print(f"{'═' * 50}")

        # 评估BC
        print("  评估 BC 策略...")
        bc_eval = RLEvaluator(bc_checkpoint, str(self.device))
        bc_result = bc_eval.evaluate_single_cat(cat_id, num_episodes)

        # 评估RL（当前模型）
        print("  评估 RL 策略...")
        rl_result = self.evaluate_single_cat(cat_id, num_episodes)

        # 对比
        improvement = {
            "reward_delta": rl_result["mean_reward"] - bc_result["mean_reward"],
            "reward_pct": (rl_result["mean_reward"] / max(1.0, bc_result["mean_reward"]) - 1) * 100,
            "trust_delta": rl_result["mean_final_trust"] - bc_result["mean_final_trust"],
            "stress_delta": rl_result["mean_final_stress"] - bc_result["mean_final_stress"],
        }

        print(f"\n  奖励: BC={bc_result['mean_reward']:.2f} → RL={rl_result['mean_reward']:.2f} "
              f"(Δ={improvement['reward_delta']:+.2f}, {improvement['reward_pct']:+.1f}%)")
        print(f"  最终信任: BC={bc_result['mean_final_trust']:.0f} → RL={rl_result['mean_final_trust']:.0f}")
        print(f"  最终压力: BC={bc_result['mean_final_stress']:.0f} → RL={rl_result['mean_final_stress']:.0f}")

        return {"bc": bc_result, "rl": rl_result, "improvement": improvement}

    def print_report(self, result: Dict):
        """打印评估报告"""
        print(f"\n{'═' * 50}")
        print(f"  评估报告: {result.get('cat_id', 'multi')}")
        print(f"{'═' * 50}")
        print(f"  Episodes: {result.get('num_episodes', 'N/A')}")
        print(f"  平均奖励: {result.get('mean_reward', 0):.2f} ± {result.get('std_reward', 0):.2f}")
        print(f"  最大奖励: {result.get('max_reward', 0):.2f}")
        print(f"  最小奖励: {result.get('min_reward', 0):.2f}")
        print(f"  最终信任: {result.get('mean_final_trust', 0):.1f}")
        print(f"  最终压力: {result.get('mean_final_stress', 0):.1f}")

        intent_dist = result.get("intent_distribution", {})
        if intent_dist:
            total = sum(intent_dist.values())
            print(f"\n  意图分布 (Top 8):")
            for intent, count in sorted(intent_dist.items(),
                                         key=lambda x: x[1], reverse=True)[:8]:
                print(f"    {intent:22s}: {count:4d} ({count/total*100:5.1f}%)")


def compute_training_curves(history_path: str) -> Dict:
    """
    从训练历史JSON中提取训练曲线数据。

    返回: {steps, rewards, policy_loss, value_loss, entropy}
    """
    with open(history_path, 'r') as f:
        data = json.load(f)

    metrics = data.get("metrics", [])
    steps = [m["step"] for m in metrics]
    p_loss = [m["policy_loss"] for m in metrics]
    v_loss = [m["value_loss"] for m in metrics]
    entropy = [m["entropy"] for m in metrics]

    return {
        "steps": steps,
        "policy_loss": p_loss,
        "value_loss": v_loss,
        "entropy": entropy,
        "episode_rewards": data.get("episode_rewards", []),
    }
