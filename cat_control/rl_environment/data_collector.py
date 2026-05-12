"""
《猫语心声》 —— 训练数据收集器

从沙盒模拟中收集 (state, action, reward, next_state, done) 轨迹，
用于阶段A：行为克隆预热，以及阶段B：PPO离线训练。

核心机制：延迟缓冲
  每 tick 记录当前 (state, action, reward) 为"待完成"。
  下一 tick 时，用新的 state 作为上一 tick 的 next_state 完成转移。

输出格式兼容 stable-baselines3 和自研PPO训练框架。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import json
import os
from datetime import datetime

from .config import INTENT_LIST, STATE_DIM


@dataclass
class Transition:
    """单步转移（state, action, reward, next_state, done）"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "state": self.state.tolist(),
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state.tolist(),
            "done": self.done,
            "info": self.info,
        }

    def is_valid(self) -> bool:
        """检查转移数据是否有效"""
        if self.state is None or self.next_state is None:
            return False
        if self.state.shape != (STATE_DIM,) or self.next_state.shape != (STATE_DIM,):
            return False
        if np.any(np.isnan(self.state)) or np.any(np.isnan(self.next_state)):
            return False
        if self.action < 0 or self.action >= len(INTENT_LIST):
            return False
        return True


@dataclass
class Episode:
    """一个episode（从开始到done）"""
    transitions: List[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    episode_id: int = 0
    cat_id: str = ""

    def add(self, t: Transition):
        if t.is_valid():
            self.transitions.append(t)
            self.total_reward += t.reward
        else:
            raise ValueError(f"Invalid transition in episode {self.episode_id}")

    def __len__(self) -> int:
        return len(self.transitions)


class DataCollector:
    """
    数据收集器 —— 使用延迟缓冲机制收集 RL 训练数据。

    两种收集模式：
    1. 行为克隆模式："bc" — 只收集 (state, action) 对
    2. RL模式："rl" — 收集完整的 (s, a, r, s', done) 转移

    延迟缓冲工作流程：
      tick T:   cat.process_interaction() → 获得 intent, reward
                → collector.record_pending(cat_id, state_T, intent, reward, info)
                → 内部存入 _pending[cat_id]

      tick T+1: cat.process_interaction() → 获得新的 state_{T+1}
                → collector.complete_pending(cat_id, state_{T+1}, done)
                → 内部用 state_{T+1} 作为 pending 的 next_state，创建完整 Transition
                → 同时把当前新 state 存入 pending 等待下一轮
    """

    def __init__(self, save_dir: str = "./training_data",
                 mode: str = "bc"):
        """
        mode: "bc" (行为克隆) 或 "rl" (RL训练)
        """
        self.save_dir = save_dir
        self.mode = mode
        self.episodes: List[Episode] = []
        self.current_episode: Optional[Episode] = None
        self.total_transitions: int = 0
        self.valid_transitions: int = 0
        self.skipped_transitions: int = 0

        # ── 延迟缓冲：{cat_id: {state, action, reward, info}} ──
        self._pending: Dict[str, Dict[str, Any]] = {}

        # 统计
        self.action_counts: Dict[int, int] = {i: 0 for i in range(len(INTENT_LIST))}
        self.reward_history: List[float] = []
        self.episode_rewards: List[float] = []

        os.makedirs(save_dir, exist_ok=True)

    # ==================== Episode 管理 ====================

    def start_episode(self, cat_id: str = ""):
        """开始一个新episode"""
        episode_id = len(self.episodes)
        self.current_episode = Episode(
            episode_id=episode_id,
            cat_id=cat_id,
        )
        self.episodes.append(self.current_episode)

    def end_episode(self):
        """结束当前episode"""
        if self.current_episode and self.current_episode.transitions:
            self.episode_rewards.append(self.current_episode.total_reward)
        # 清除该 episode 相关的 pending
        self._pending.clear()
        self.current_episode = None

    # ==================== 延迟缓冲机制（核心） ====================

    def record_pending(self, cat_id: str, state: np.ndarray,
                       intent: str, reward: float,
                       info: Dict = None):
        """
        tick T：记录当前 state 和选择的 action，等待下一 tick 的 next_state。

        参数:
            cat_id: 猫咪ID
            state: 当前状态向量 (形状 STATE_DIM,)
            intent: 选择的宏观意图
            reward: 即时奖励
            info: 附加信息
        """
        action_idx = INTENT_LIST.index(intent) if intent in INTENT_LIST else 0
        self._pending[cat_id] = {
            "state": state.astype(np.float32),
            "action": action_idx,
            "reward": float(reward),
            "info": info or {},
        }

    def complete_pending(self, cat_id: str, next_state: np.ndarray,
                        done: bool = False) -> Optional[Transition]:
        """
        tick T+1：用当前状态作为上一 tick pending 的 next_state，完成转移。

        返回完成的 Transition，若该 cat 无 pending 则返回 None。
        """
        if cat_id not in self._pending:
            return None

        pending = self._pending.pop(cat_id)

        try:
            t = Transition(
                state=pending["state"].astype(np.float32),
                action=pending["action"],
                reward=pending["reward"],
                next_state=next_state.astype(np.float32),
                done=done,
                info=pending["info"],
            )
        except (ValueError, TypeError):
            self.skipped_transitions += 1
            return None

        if not t.is_valid():
            self.skipped_transitions += 1
            return None

        # 统计
        self.action_counts[t.action] += 1
        self.total_transitions += 1
        self.valid_transitions += 1
        self.reward_history.append(t.reward)

        if self.current_episode is None:
            self.start_episode(cat_id)
        self.current_episode.add(t)

        return t

    def flush_pending(self, dummy_next_state: np.ndarray = None):
        """清空所有 pending（用于模拟结束时）"""
        if dummy_next_state is None:
            dummy_next_state = np.zeros(STATE_DIM, dtype=np.float32)

        for cat_id in list(self._pending.keys()):
            self.complete_pending(cat_id, dummy_next_state, done=True)
        self._pending.clear()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    # ==================== 数据导出 ====================

    def export_bc_data(self, filename: str = None) -> str:
        """
        导出行为克隆数据：(state, action)
        格式：.npz 压缩文件，包含 states 和 actions 数组
        """
        states = []
        actions = []

        for ep in self.episodes:
            for t in ep.transitions:
                if t.is_valid():
                    states.append(t.state)
                    actions.append(t.action)

        if not states:
            print("[DataCollector] 警告：没有有效数据可导出！")
            return ""

        states_arr = np.stack(states, axis=0).astype(np.float32)
        actions_arr = np.array(actions, dtype=np.int32)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bc_data_{len(states)}samples_{timestamp}.npz"

        filepath = os.path.join(self.save_dir, filename)
        np.savez_compressed(filepath,
                           states=states_arr,
                           actions=actions_arr,
                           num_samples=len(states),
                           intent_list=np.array(INTENT_LIST),
                           state_dim=STATE_DIM)
        print(f"\n[DataCollector] 行为克隆数据已导出: {filepath}")
        print(f"  有效样本数: {len(states)}, 状态维度: {STATE_DIM}")
        print(f"  文件大小: {os.path.getsize(filepath) / 1024:.1f} KB")
        return filepath

    def export_rl_data(self, filename: str = None) -> str:
        """
        导出RL训练数据：完整轨迹 JSON
        """
        episodes_data = []
        for ep in self.episodes:
            if not ep.transitions:
                continue
            ep_data = {
                "episode_id": ep.episode_id,
                "cat_id": ep.cat_id,
                "total_reward": ep.total_reward,
                "num_transitions": len(ep.transitions),
                "transitions": [t.to_dict() for t in ep.transitions],
            }
            episodes_data.append(ep_data)

        if not episodes_data:
            print("[DataCollector] 警告：没有有效Episode可导出！")
            return ""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rl_trajectories_{self.valid_transitions}steps_{timestamp}.json"

        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "num_episodes": len(episodes_data),
                "total_transitions": self.total_transitions,
                "valid_transitions": self.valid_transitions,
                "intent_list": INTENT_LIST,
                "state_dim": STATE_DIM,
                "episodes": episodes_data,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n[DataCollector] RL轨迹数据已导出: {filepath}")
        print(f"  Episodes: {len(episodes_data)}, Transitions: {self.total_transitions}")
        return filepath

    def export_csv(self, filename: str = None) -> str:
        """导出为CSV格式（方便分析）"""
        import csv

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{self.valid_transitions}steps_{timestamp}.csv"

        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "step", "cat_id", "state_mean", "state_std",
                           "action_idx", "intent", "reward", "done",
                           "next_state_mean", "next_state_std"])
            for ep in self.episodes:
                for i, t in enumerate(ep.transitions):
                    writer.writerow([
                        ep.episode_id, i, ep.cat_id,
                        round(float(np.mean(t.state)), 6),
                        round(float(np.std(t.state)), 6),
                        t.action, INTENT_LIST[t.action],
                        round(t.reward, 4), t.done,
                        round(float(np.mean(t.next_state)), 6),
                        round(float(np.std(t.next_state)), 6),
                    ])
        print(f"[DataCollector] CSV数据已导出: {filepath}")
        return filepath

    # ==================== 统计报告 ====================

    def stats_report(self) -> str:
        """生成统计报告"""
        total = sum(self.action_counts.values()) or 1
        sorted_actions = sorted(
            self.action_counts.items(),
            key=lambda x: -x[1]
        )

        lines = [
            "=" * 60,
            "  训练数据收集统计",
            "=" * 60,
            f"  总Episodes: {len(self.episodes)}",
            f"  总Transitions: {self.total_transitions}",
            f"  有效Transitions: {self.valid_transitions}",
            f"  跳过Transitions: {self.skipped_transitions}",
            f"  总Reward: {sum(self.reward_history):.2f}",
            f"  平均Reward/step: {np.mean(self.reward_history):.4f}" if self.reward_history else "",
            f"  平均Reward/episode: {np.mean(self.episode_rewards):.2f}" if self.episode_rewards else "",
            "",
            "  意图分布:",
        ]

        for action_idx, count in sorted_actions:
            intent = INTENT_LIST[action_idx]
            bar = "█" * int(count / total * 40)
            lines.append(f"    {intent:25s} {count:6d} ({count/total:5.1%}) {bar}")

        # 意图覆盖检查
        unused = [INTENT_LIST[i] for i, c in self.action_counts.items() if c == 0]
        if unused:
            lines.append(f"\n  ⚠ 未收集到的意图 ({len(unused)}个): {', '.join(unused)}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def progress_summary(self) -> str:
        """简要进度摘要（用于长时间运行的进度打印）"""
        return (f"已收集 {self.valid_transitions} 条有效转移, "
                f"跳过 {self.skipped_transitions}, "
                f"pending {self.pending_count}")

    def reset(self):
        """重置所有数据"""
        self.episodes.clear()
        self.current_episode = None
        self._pending.clear()
        self.total_transitions = 0
        self.valid_transitions = 0
        self.skipped_transitions = 0
        self.action_counts = {i: 0 for i in range(len(INTENT_LIST))}
        self.reward_history.clear()
        self.episode_rewards.clear()
