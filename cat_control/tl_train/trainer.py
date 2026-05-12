"""
《猫语心声》 —— PPO训练编排器

支持两种训练模式：
1. single_cat: 单猫PPO训练（验证基本需求学习）
2. self_play: 3猫自对弈训练（共享策略+性格条件化）
"""

from __future__ import annotations
import sys
import os
import time
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_environment.config import INTENT_LIST, PERSONALITY_DIM
from rl_environment.cat_state import MemoryManager
from rl_train.config import TrainConfig, CAT_PERSONALITIES
from rl_train.env_wrapper import SingleCatEnv, MultiCatEnv
from rl_train.ppo import PPO


class PPOTrainer:
    """PPO训练编排器"""

    def __init__(self, config: TrainConfig = None):
        self.config = config or TrainConfig()
        self.cfg = self.config

        # PPO训练器（支持多猫共享策略网络）
        self.ppo = PPO(
            state_dim=self.cfg.state_dim,
            personality_dim=self.cfg.personality_dim,
            intent_num=self.cfg.intent_num,
            embed_dim=self.cfg.embed_dim,
            nhead=self.cfg.nhead,
            ff_dim=self.cfg.ff_dim,
            num_layers=self.cfg.num_layers,
            seq_len=self.cfg.seq_len,
            dropout=self.cfg.dropout,
            lr=self.cfg.learning_rate,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            clip_epsilon=self.cfg.clip_epsilon,
            value_loss_coef=self.cfg.value_loss_coef,
            entropy_coef=self.cfg.entropy_coef,
            max_grad_norm=self.cfg.max_grad_norm,
            ppo_epochs=self.cfg.ppo_epochs,
            batch_size=self.cfg.batch_size,
            target_kl=self.cfg.target_kl,
            steps_per_rollout=self.cfg.steps_per_rollout,
            device=self.cfg.device,
        )

        # 状态序列缓冲区（每只猫维护一个独立的序列）
        self._state_seqs: Dict[str, deque] = {}

        # 训练历史
        self.history: List[Dict] = []

    def _get_state_seq(self, cat_id: str, obs: np.ndarray) -> np.ndarray:
        """维护并返回状态序列 [seq_len, state_dim]"""
        if cat_id not in self._state_seqs:
            self._state_seqs[cat_id] = deque(
                [np.zeros(self.cfg.state_dim, dtype=np.float32)] * self.cfg.seq_len,
                maxlen=self.cfg.seq_len
            )
        self._state_seqs[cat_id].append(obs)
        return np.stack(list(self._state_seqs[cat_id]), axis=0)

    def _reset_state_seq(self, cat_id: str):
        """重置某只猫的状态序列（新episode开始）"""
        self._state_seqs[cat_id] = deque(
            [np.zeros(self.cfg.state_dim, dtype=np.float32)] * self.cfg.seq_len,
            maxlen=self.cfg.seq_len
        )

    # ═══════════════════════════════════════════
    #  单猫训练
    # ═══════════════════════════════════════════

    def train_single_cat(self, cat_id: str = "oreo",
                        total_timesteps: int = None) -> Dict:
        """
        单猫PPO训练 —— 验证RL能够学会基本需求（饥饿→进食、疲劳→睡眠等）。
        """
        total_timesteps = total_timesteps or self.cfg.total_timesteps
        save_dir = self.cfg.model_save_dir
        cat_name = CAT_PERSONALITIES.get(cat_id, {})

        print(f"\n{'═' * 60}")
        print(f"  单猫PPO训练: {cat_id}")
        print(f"  总步数: {total_timesteps:,}")
        print(f"{'═' * 60}")

        # 加载BC预训练权重
        if self.cfg.bc_checkpoint:
            self.ppo.load_bc_weights(self.cfg.bc_checkpoint)

        env = SingleCatEnv(cat_id=cat_id, max_steps_per_episode=self.cfg.steps_per_rollout)
        personality = env.get_personality()

        obs = env.reset()
        self._reset_state_seq(cat_id)

        episode_rewards = []
        episode_lengths = []
        current_ep_reward = 0.0
        current_ep_len = 0
        best_mean_reward = -float("inf")

        start_time = time.time()
        last_log_time = start_time

        for step in range(total_timesteps):
            # 构造状态序列
            state_seq = self._get_state_seq(cat_id, obs)

            # 选择动作
            action, log_prob, value = self.ppo.select_action(state_seq, personality)

            # 执行
            next_obs, reward, terminated, truncated, info = env.step(action)

            # 存储
            self.ppo.buffer.store(obs, personality, action, reward, terminated or truncated,
                                 value, log_prob)
            self.ppo.total_steps += 1

            obs = next_obs
            current_ep_reward += reward
            current_ep_len += 1

            # Episode结束
            if terminated or truncated:
                episode_rewards.append(current_ep_reward)
                episode_lengths.append(current_ep_len)
                current_ep_reward = 0.0
                current_ep_len = 0
                obs = env.reset()
                self._reset_state_seq(cat_id)

            # PPO更新
            if len(self.ppo.buffer) >= self.cfg.steps_per_rollout:
                metrics = self.ppo.update()
                metrics["step"] = step + 1
                self.history.append(metrics)

                # 日志
                if (step + 1) % self.cfg.log_freq < self.cfg.steps_per_rollout:
                    elapsed = time.time() - start_time
                    fps = (step + 1) / elapsed if elapsed > 0 else 0
                    avg_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0
                    print(f"[Step {step+1:>8,}] "
                          f"avg_ep_reward: {avg_reward:>7.2f} | "
                          f"policy_loss: {metrics['policy_loss']:.4f} | "
                          f"value_loss: {metrics['value_loss']:.4f} | "
                          f"entropy: {metrics['entropy']:.4f} | "
                          f"kl: {metrics.get('kl', 0):.5f} | "
                          f"fps: {fps:.0f}")

            # 定期保存
            if (step + 1) % self.cfg.save_freq == 0:
                ckpt_path = os.path.join(save_dir, f"ppo_single_{cat_id}_step{step+1}.pt")
                mean_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0
                self.ppo.save(ckpt_path, extra={
                    "cat_id": cat_id, "mean_reward_20ep": mean_reward,
                    "episode_rewards": episode_rewards,
                })
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    best_path = os.path.join(save_dir, f"ppo_single_{cat_id}_best.pt")
                    self.ppo.save(best_path, extra={"mean_reward_20ep": mean_reward})

        # 最终保存
        final_path = os.path.join(save_dir, f"ppo_single_{cat_id}_final.pt")
        self.ppo.save(final_path)

        # 导出ONNX
        onnx_path = os.path.join(self.cfg.export_dir, f"ppo_single_{cat_id}.onnx")
        self.ppo.export_onnx(onnx_path)

        # 保存训练历史
        history_path = os.path.join(self.cfg.log_dir, f"train_history_single_{cat_id}.json")
        with open(history_path, 'w') as f:
            json.dump({
                "cat_id": cat_id,
                "total_steps": total_timesteps,
                "episode_rewards": [float(r) for r in episode_rewards],
                "episode_lengths": episode_lengths,
                "best_mean_reward": float(best_mean_reward),
                "metrics": [{k: float(v) if isinstance(v, (np.floating,)) else v
                           for k, v in m.items()} for m in self.history],
            }, f, indent=2)

        elapsed = time.time() - start_time
        print(f"\n  ✓ 单猫训练完成 ({elapsed:.0f}s, {total_timesteps:,} steps)")
        print(f"  最佳20ep平均奖励: {best_mean_reward:.2f}")
        return {"best_mean_reward": best_mean_reward, "episode_rewards": episode_rewards}

    # ═══════════════════════════════════════════
    #  多猫自对弈训练
    # ═══════════════════════════════════════════

    def train_self_play(self, cat_ids: List[str] = None,
                       total_timesteps: int = None) -> Dict:
        """
        3猫自对弈训练 —— 共享策略网络 + 独立性格嵌入。

        所有猫共用同一个策略网络，通过FiLM的性格条件化产生差异化行为。
        在共享环境中竞争资源（食物、玩具、藏身处、玩家关注）。
        """
        cat_ids = cat_ids or list(CAT_PERSONALITIES.keys())
        total_timesteps = total_timesteps or self.cfg.total_timesteps
        save_dir = self.cfg.model_save_dir

        print(f"\n{'═' * 60}")
        print(f"  多猫自对弈PPO训练")
        print(f"  猫咪: {cat_ids}")
        print(f"  总步数: {total_timesteps:,}")
        print(f"  策略共享: {'是' if self.cfg.cats_share_policy else '否'}")
        print(f"{'═' * 60}")

        # 加载BC预训练权重
        if self.cfg.bc_checkpoint:
            self.ppo.load_bc_weights(self.cfg.bc_checkpoint)

        env = MultiCatEnv(cat_ids=cat_ids, max_steps_per_episode=self.cfg.steps_per_rollout)
        personalities = env.get_personalities()

        obs_dict = env.reset()
        for cid in cat_ids:
            self._reset_state_seq(cid)

        episode_rewards = {cid: [] for cid in cat_ids}
        episode_lengths = []
        current_ep_rewards = {cid: 0.0 for cid in cat_ids}
        current_ep_len = 0
        best_total_reward = -float("inf")

        start_time = time.time()

        for step in range(total_timesteps):
            intent_dict = {}
            log_prob_dict = {}
            value_dict = {}

            # 每只猫独立决策
            for cid in cat_ids:
                state_seq = self._get_state_seq(cid, obs_dict[cid])
                action, log_prob, value = self.ppo.select_action(
                    state_seq, personalities[cid]
                )
                intent_dict[cid] = action
                log_prob_dict[cid] = log_prob
                value_dict[cid] = value

            # 所有猫同时执行
            next_obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(intent_dict)

            # 存储每条轨迹
            for cid in cat_ids:
                done = term_dict[cid] or trunc_dict[cid]
                self.ppo.buffer.store(
                    obs_dict[cid], personalities[cid],
                    intent_dict[cid], reward_dict[cid], done,
                    value_dict[cid], log_prob_dict[cid]
                )
                self.ppo.total_steps += 1
                current_ep_rewards[cid] += reward_dict[cid]

            obs_dict = next_obs_dict
            current_ep_len += 1

            # Episode结束
            all_done = all(term_dict[cid] or trunc_dict[cid] for cid in cat_ids)
            if all_done:
                total_ep_reward = sum(current_ep_rewards.values())
                episode_lengths.append(current_ep_len)
                for cid in cat_ids:
                    episode_rewards[cid].append(current_ep_rewards[cid])
                    current_ep_rewards[cid] = 0.0
                current_ep_len = 0
                obs_dict = env.reset()
                for cid in cat_ids:
                    self._reset_state_seq(cid)

            # PPO更新（使用所有猫的混合数据）
            if len(self.ppo.buffer) >= self.cfg.steps_per_rollout:
                metrics = self.ppo.update()
                metrics["step"] = step + 1
                self.history.append(metrics)

                if (step + 1) % self.cfg.log_freq < self.cfg.steps_per_rollout:
                    elapsed = time.time() - start_time
                    fps = (step + 1) / elapsed if elapsed > 0 else 0
                    avg_rew = {cid: np.mean(episode_rewards[cid][-10:]) if episode_rewards[cid] else 0
                              for cid in cat_ids}
                    total_avg = sum(avg_rew.values())
                    rew_str = " | ".join(f"{cid}:{avg_rew[cid]:.1f}" for cid in cat_ids)
                    print(f"[Step {step+1:>8,}] "
                          f"total_reward: {total_avg:>6.1f} ({rew_str}) | "
                          f"p_loss: {metrics['policy_loss']:.4f} | "
                          f"v_loss: {metrics['value_loss']:.4f} | "
                          f"H: {metrics['entropy']:.4f} | "
                          f"fps: {fps:.0f}")

            # 定期保存
            if (step + 1) % self.cfg.save_freq == 0:
                total_avg = sum(
                    np.mean(episode_rewards[cid][-10:]) if episode_rewards[cid] else 0
                    for cid in cat_ids
                )
                ckpt_path = os.path.join(save_dir, f"ppo_selfplay_step{step+1}.pt")
                self.ppo.save(ckpt_path, extra={
                    "cat_ids": cat_ids,
                    "total_reward_10ep": total_avg,
                    "episode_rewards": {cid: [float(r) for r in rews]
                                      for cid, rews in episode_rewards.items()},
                })
                if total_avg > best_total_reward:
                    best_total_reward = total_avg
                    best_path = os.path.join(save_dir, "ppo_selfplay_best.pt")
                    self.ppo.save(best_path, extra={"total_reward_10ep": total_avg})

        # 最终保存
        final_path = os.path.join(save_dir, "ppo_selfplay_final.pt")
        self.ppo.save(final_path)

        # 导出ONNX
        onnx_path = os.path.join(self.cfg.export_dir, "ppo_selfplay.onnx")
        self.ppo.export_onnx(onnx_path)

        # 保存训练历史
        history_path = os.path.join(self.cfg.log_dir, "train_history_selfplay.json")
        with open(history_path, 'w') as f:
            json.dump({
                "cat_ids": cat_ids,
                "total_steps": total_timesteps,
                "best_total_reward": float(best_total_reward),
                "episode_rewards": {cid: [float(r) for r in rews]
                                  for cid, rews in episode_rewards.items()},
                "episode_lengths": episode_lengths,
                "metrics": [{k: float(v) if isinstance(v, (np.floating,)) else v
                           for k, v in m.items()} for m in self.history],
            }, f, indent=2)

        elapsed = time.time() - start_time
        print(f"\n  ✓ 自对弈训练完成 ({elapsed:.0f}s, {total_timesteps:,} steps)")
        print(f"  最佳10ep总奖励: {best_total_reward:.2f}")
        return {"best_total_reward": best_total_reward, "episode_rewards": episode_rewards}

    # ═══════════════════════════════════════════
    #  完整训练流水线（BC→单猫→自对弈）
    # ═══════════════════════════════════════════

    def train_full_pipeline(self,
                           single_cat_timesteps: int = 500_000,
                           self_play_timesteps: int = 1_500_000) -> Dict:
        """
        完整训练流水线：BC预训练 → 单猫PPO → 多猫自对弈
        """
        print(f"\n{'═' * 60}")
        print(f"  《猫语心声》RL策略网络 — 完整训练流水线")
        print(f"  BC预热 → 单猫PPO (500k) → 自对弈 (1.5M)")
        print(f"{'═' * 60}")

        results = {}

        # 阶段1: BC预热（已在rl_clone中完成，这里只需加载）
        print(f"\n[阶段1] BC预热权重加载")
        if self.cfg.bc_checkpoint:
            self.ppo.load_bc_weights(self.cfg.bc_checkpoint)
        else:
            print("  ⚠ 未找到BC检查点，从头开始训练")

        # 阶段2: 单猫PPO（用奥利奥——性格最均衡）
        print(f"\n{'─' * 60}")
        print(f"[阶段2] 单猫PPO训练 — 验证基本需求学习")
        print(f"{'─' * 60}")
        results["single_cat"] = self.train_single_cat(
            cat_id="oreo", total_timesteps=single_cat_timesteps
        )

        # 阶段3: 多猫自对弈
        print(f"\n{'─' * 60}")
        print(f"[阶段3] 3猫自对弈训练 — 涌现社交/竞争行为")
        print(f"{'─' * 60}")
        results["self_play"] = self.train_self_play(
            cat_ids=["xiaoxue", "oreo", "orange"],
            total_timesteps=self_play_timesteps,
        )

        print(f"\n{'═' * 60}")
        print(f"  ✓ 完整训练流水线完成！")
        print(f"{'═' * 60}")

        return results
