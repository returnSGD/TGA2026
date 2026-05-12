"""
《猫语心声》 —— PPO算法实现

基于技术策划案v2 §4.1.2 的策略网络 + 标准PPO-CLIP + GAE。
支持性格条件化（FiLM调制），单猫/多猫自对弈训练。
"""

from __future__ import annotations
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Dict, List, Optional
from collections import deque
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_clone.model import RLPolicyNetwork


class RolloutBuffer:
    """
    PPO Rollout缓冲区 — 收集训练轨迹。

    存储 (state_seq, personality, action, reward, done, value, log_prob)
    用于GAE优势计算和PPO更新。
    """

    def __init__(self, state_dim: int, personality_dim: int,
                 seq_len: int, capacity: int):
        self.state_dim = state_dim
        self.personality_dim = personality_dim
        self.seq_len = seq_len
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.states = np.zeros((self.capacity, self.seq_len, self.state_dim), dtype=np.float32)
        self.personalities = np.zeros((self.capacity, self.personality_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.log_probs = np.zeros(self.capacity, dtype=np.float32)
        self.ptr = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.ptr

    def add(self, state_seq: np.ndarray, personality: np.ndarray,
            action: int, reward: float, done: bool,
            value: float, log_prob: float):
        """添加一条转移"""
        idx = self.ptr
        self.states[idx] = state_seq
        self.personalities[idx] = personality
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.values[idx] = value
        self.log_probs[idx] = log_prob

        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def get_all(self) -> Dict[str, np.ndarray]:
        """返回全部数据"""
        n = len(self)
        return {
            "states": self.states[:n],
            "personalities": self.personalities[:n],
            "actions": self.actions[:n],
            "rewards": self.rewards[:n],
            "dones": self.dones[:n],
            "values": self.values[:n],
            "log_probs": self.log_probs[:n],
        }


class PPOBuffer:
    """
    带序列记忆的PPO缓冲区。

    维护一个状态历史队列（seq_len步），每次添加时自动构造 (seq_len, state_dim) 的序列。
    """

    def __init__(self, state_dim: int, personality_dim: int,
                 seq_len: int, capacity: int):
        self.state_dim = state_dim
        self.personality_dim = personality_dim
        self.seq_len = seq_len
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.ptr = 0
        self.episode_start = True
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.personalities = np.zeros((self.capacity, self.personality_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.log_probs = np.zeros(self.capacity, dtype=np.float32)
        self.state_buffer = deque(maxlen=self.seq_len)

    def __len__(self) -> int:
        return self.ptr

    def store(self, state: np.ndarray, personality: np.ndarray,
              action: int, reward: float, done: bool,
              value: float, log_prob: float):
        """存储单步转移"""
        idx = self.ptr
        self.states[idx] = state
        self.personalities[idx] = personality
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float,
                    gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
        """计算GAE优势和回报"""
        n = len(self)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(n)):
            next_value = self.values[t + 1] if t + 1 < n else last_value
            next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae

            advantages[t] = gae
            returns[t] = gae + self.values[t]

        return advantages, returns

    def get_batch(self, batch_indices: np.ndarray, seq_len: int) -> Dict[str, torch.Tensor]:
        """获取按索引的批次数据，自动构造状态序列"""
        batch_states_seq = []
        for i in batch_indices:
            seq = []
            for offset in range(seq_len - 1, -1, -1):
                if i - offset >= 0 and i - offset < len(self):
                    seq.append(self.states[i - offset])
                else:
                    seq.append(np.zeros(self.state_dim, dtype=np.float32))
            batch_states_seq.append(np.stack(seq, axis=0))

        return {
            "states_seq": torch.tensor(np.stack(batch_states_seq, axis=0), dtype=torch.float32),
            "personalities": torch.tensor(self.personalities[batch_indices], dtype=torch.float32),
            "actions": torch.tensor(self.actions[batch_indices], dtype=torch.long),
            "advantages": torch.tensor(
                self._advantages[batch_indices] if hasattr(self, '_advantages') else np.zeros(len(batch_indices)),
                dtype=torch.float32),
            "returns": torch.tensor(
                self._returns[batch_indices] if hasattr(self, '_returns') else np.zeros(len(batch_indices)),
                dtype=torch.float32),
            "old_log_probs": torch.tensor(self.log_probs[batch_indices], dtype=torch.float32),
            "old_values": torch.tensor(self.values[batch_indices], dtype=torch.float32),
        }

    def set_advantages_returns(self, advantages: np.ndarray, returns: np.ndarray):
        self._advantages = advantages
        self._returns = returns


class PPO:
    """
    PPO (Proximal Policy Optimization) 训练器。

    支持：
    - 性格条件化策略（FiLM调制）
    - GAE优势估计
    - PPO-CLIP目标
    - 值函数裁剪
    - 熵正则化
    - KL早停
    - BC预训练权重加载
    """

    def __init__(self, state_dim: int = 422, personality_dim: int = 8,
                 intent_num: int = 15, embed_dim: int = 128,
                 nhead: int = 4, ff_dim: int = 256, num_layers: int = 3,
                 seq_len: int = 4, dropout: float = 0.1,
                 lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_epsilon: float = 0.2,
                 value_loss_coef: float = 0.5, entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5, ppo_epochs: int = 10,
                 batch_size: int = 64, target_kl: float = 0.015,
                 steps_per_rollout: int = 2048,
                 device: str = "cuda"):
        self.state_dim = state_dim
        self.personality_dim = personality_dim
        self.intent_num = intent_num
        self.seq_len = seq_len
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl
        self.steps_per_rollout = steps_per_rollout
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 策略网络
        self.policy = RLPolicyNetwork(
            state_dim=state_dim, embed_dim=embed_dim,
            num_intents=intent_num, seq_len=seq_len,
            personality_dim=personality_dim,
            nhead=nhead, ff_dim=ff_dim, num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        # 缓冲区
        self.buffer = PPOBuffer(state_dim, personality_dim, seq_len, steps_per_rollout)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=lr, weight_decay=1e-5
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=2_000_000 // steps_per_rollout, eta_min=1e-5
        )

        # 统计
        self.total_steps = 0
        self.total_updates = 0
        self.best_reward = -float("inf")
        self.train_metrics: List[Dict] = []

    def load_bc_weights(self, checkpoint_path: str) -> bool:
        """
        加载BC预训练权重作为PPO初始策略。

        返回True if successful, else False。
        """
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"[PPO] BC检查点不存在: {checkpoint_path}，从头开始训练")
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_state = checkpoint.get("model_state_dict", checkpoint)

        # 过滤不匹配的键（BC的seq_len=1, PPO的seq_len=4）
        policy_dict = self.policy.state_dict()
        filtered_state = {}
        skipped = []
        for k, v in model_state.items():
            if k in policy_dict:
                if v.shape == policy_dict[k].shape:
                    filtered_state[k] = v
                else:
                    skipped.append(k)

        if skipped:
            print(f"[PPO] BC权重跳过 {len(skipped)} 个形状不匹配的键: {skipped[:5]}...")

        self.policy.load_state_dict(filtered_state, strict=False)
        print(f"[PPO] 成功加载BC预训练权重: {checkpoint_path}")
        print(f"  匹配 {len(filtered_state)}/{len(policy_dict)} 个参数")
        return True

    def select_action(self, state_seq: np.ndarray, personality: np.ndarray,
                      deterministic: bool = False) -> Tuple[int, float, float]:
        """
        选择动作。

        参数:
            state_seq: [seq_len, state_dim]
            personality: [personality_dim]
        返回:
            action_idx, log_prob, value
        """
        with torch.no_grad():
            s = torch.from_numpy(state_seq).float().unsqueeze(0).to(self.device)  # [1, S, D]
            p = torch.from_numpy(personality).float().unsqueeze(0).to(self.device)  # [1, P]

            logits, value = self.policy(s, p)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)

            if deterministic:
                action = torch.argmax(probs, dim=-1).item()
            else:
                action = dist.sample().item()

            log_prob = dist.log_prob(torch.tensor(action, device=self.device))

        return action, log_prob.item(), value.item()

    def evaluate_actions(self, states_seq: torch.Tensor,
                        personalities: torch.Tensor,
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量评估动作（用于PPO更新）。

        返回:
            log_probs, values, entropy
        """
        logits, values = self.policy(states_seq, personalities)  # [B, 15], [B]
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)  # [B]
        entropy = dist.entropy().mean()     # scalar

        return log_probs, values, entropy

    def update(self) -> Dict:
        """执行一轮PPO更新，返回指标"""
        n = len(self.buffer)
        if n < self.batch_size:
            return {"error": f"缓冲不足: {n} < {self.batch_size}"}

        # 计算GAE
        last_state_seq = self._get_last_seq()
        last_personality = self.buffer.personalities[n - 1] if n > 0 else np.zeros(self.personality_dim)
        with torch.no_grad():
            _, last_value = self.policy(
                torch.from_numpy(last_state_seq).float().unsqueeze(0).to(self.device),
                torch.from_numpy(last_personality).float().unsqueeze(0).to(self.device),
            )
            last_val = last_value.item()
        last_val = last_val if not self.buffer.dones[n - 1] else 0.0

        advantages, returns = self.buffer.compute_gae(last_val, self.gamma, self.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.set_advantages_returns(advantages, returns)

        # PPO多epoch更新
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_batches = 0

        for epoch in range(self.ppo_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                batch = self.buffer.get_batch(batch_idx, self.seq_len)

                states_seq = batch["states_seq"].to(self.device)
                personalities = batch["personalities"].to(self.device)
                actions = batch["actions"].to(self.device)
                advantages_b = batch["advantages"].to(self.device)
                returns_b = batch["returns"].to(self.device)
                old_log_probs = batch["old_log_probs"].to(self.device)
                old_values = batch["old_values"].to(self.device)

                # 评估当前策略
                new_log_probs, new_values, entropy = self.evaluate_actions(
                    states_seq, personalities, actions
                )

                # PPO-CLIP 策略损失
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                   1.0 + self.clip_epsilon) * advantages_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # 值函数损失（带裁剪）
                value_clipped = old_values + torch.clamp(
                    new_values - old_values, -self.clip_epsilon, self.clip_epsilon
                )
                value_loss_unclipped = (new_values - returns_b) ** 2
                value_loss_clipped = (value_clipped - returns_b) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # 总损失
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_batches += 1

                # KL早停
                with torch.no_grad():
                    kl = ((ratio - 1) - (new_log_probs - old_log_probs)).mean().item()
                if self.target_kl and kl > self.target_kl * 1.5:
                    break

        # 清空缓冲
        self.buffer.reset()
        self.total_updates += 1
        self.scheduler.step()

        metrics = {
            "policy_loss": total_policy_loss / max(1, n_batches),
            "value_loss": total_value_loss / max(1, n_batches),
            "entropy": total_entropy / max(1, n_batches),
            "lr": self.scheduler.get_last_lr()[0],
            "kl": kl if 'kl' in dir() else 0.0,
            "total_steps": self.total_steps,
            "total_updates": self.total_updates,
        }
        self.train_metrics.append(metrics)
        return metrics

    def _get_last_seq(self) -> np.ndarray:
        """构造缓冲区最后一个状态序列"""
        n = len(self.buffer)
        seq = []
        for offset in range(self.seq_len - 1, -1, -1):
            idx = n - 1 - offset
            if idx >= 0:
                seq.append(self.buffer.states[idx])
            else:
                seq.append(np.zeros(self.state_dim, dtype=np.float32))
        return np.stack(seq, axis=0)

    def save(self, path: str, extra: Dict = None):
        """保存模型检查点"""
        checkpoint = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "total_updates": self.total_updates,
            "best_reward": self.best_reward,
            "train_metrics": self.train_metrics,
            "config": {
                "state_dim": self.state_dim,
                "personality_dim": self.personality_dim,
                "intent_num": self.intent_num,
                "seq_len": self.seq_len,
            },
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)
        print(f"[PPO] 模型已保存: {path}")

    def load(self, path: str):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.total_updates = checkpoint.get("total_updates", 0)
        self.best_reward = checkpoint.get("best_reward", -float("inf"))
        print(f"[PPO] 模型已加载: {path} (step {self.total_steps})")

    def export_onnx(self, path: str):
        """导出ONNX格式"""
        self.policy.eval()
        dummy_state = torch.randn(1, self.seq_len, self.state_dim, device=self.device)
        dummy_personality = torch.randn(1, self.personality_dim, device=self.device)

        torch.onnx.export(
            self.policy,
            (dummy_state, dummy_personality),
            path,
            input_names=["state_seq", "personality_embed"],
            output_names=["action_logits", "state_value"],
            dynamic_axes={
                "state_seq": {0: "batch"},
                "personality_embed": {0: "batch"},
                "action_logits": {0: "batch"},
                "state_value": {0: "batch"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
        print(f"[PPO] ONNX模型已导出: {path}")
