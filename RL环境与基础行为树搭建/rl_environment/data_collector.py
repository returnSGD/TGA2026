"""
《猫语心声》 —— 训练数据收集器

从沙盒模拟中收集 (state, action, reward, next_state, done) 轨迹，
用于阶段A：行为克隆预热，以及阶段B：PPO离线训练。

输出格式兼容 stable-baselines3 和自研PPO训练框架。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
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


@dataclass
class Episode:
    """一个episode（从开始到done）"""
    transitions: List[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    episode_id: int = 0
    cat_id: str = ""

    def add(self, t: Transition):
        self.transitions.append(t)
        self.total_reward += t.reward

    def __len__(self) -> int:
        return len(self.transitions)


class DataCollector:
    """
    数据收集器 —— 从沙盒模拟中收集RL训练数据。

    两种收集模式：
    1. 行为克隆模式：只收集 (state, action) 对
    2. RL模式：收集完整的 (s, a, r, s', done) 转移
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

        # 统计
        self.action_counts: Dict[int, int] = {i: 0 for i in range(len(INTENT_LIST))}
        self.reward_history: List[float] = []
        self.episode_rewards: List[float] = []

        os.makedirs(save_dir, exist_ok=True)

    def start_episode(self, cat_id: str = ""):
        """开始一个新episode"""
        episode_id = len(self.episodes)
        self.current_episode = Episode(
            episode_id=episode_id,
            cat_id=cat_id,
        )
        self.episodes.append(self.current_episode)

    def record_step(self, state: np.ndarray, intent: str,
                    reward: float, next_state: np.ndarray,
                    done: bool = False, info: Dict = None):
        """记录单步转移"""
        if self.current_episode is None:
            self.start_episode()

        action_idx = INTENT_LIST.index(intent) if intent in INTENT_LIST else 0
        self.action_counts[action_idx] += 1
        self.total_transitions += 1

        t = Transition(
            state=state.astype(np.float32),
            action=action_idx,
            reward=float(reward),
            next_state=next_state.astype(np.float32),
            done=done,
            info=info or {},
        )
        self.current_episode.add(t)
        self.reward_history.append(reward)

    def end_episode(self):
        """结束当前episode"""
        if self.current_episode and self.current_episode.transitions:
            self.episode_rewards.append(self.current_episode.total_reward)
        self.current_episode = None

    # ==================== 数据导出 ====================

    def export_bc_data(self, filename: str = None) -> str:
        """
        导出行为克隆数据：(state, action)
        格式：{"states": [...], "actions": [...]}
        """
        states = []
        actions = []
        for ep in self.episodes:
            for t in ep.transitions:
                states.append(t.state)
                actions.append(t.action)

        data = {
            "states": np.stack(states, axis=0),
            "actions": np.array(actions, dtype=np.int32),
            "num_samples": len(states),
            "intent_list": INTENT_LIST,
            "state_dim": STATE_DIM,
        }

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bc_data_{timestamp}.npz"

        filepath = os.path.join(self.save_dir, filename)
        np.savez_compressed(filepath, **data)
        print(f"[DataCollector] 行为克隆数据已导出: {filepath}")
        print(f"  样本数: {len(states)}, 状态维度: {STATE_DIM}")
        return filepath

    def export_rl_data(self, filename: str = None) -> str:
        """
        导出RL训练数据：完整轨迹
        格式：{"episodes": [...]}
        """
        episodes_data = []
        for ep in self.episodes:
            ep_data = {
                "episode_id": ep.episode_id,
                "cat_id": ep.cat_id,
                "total_reward": ep.total_reward,
                "transitions": [t.to_dict() for t in ep.transitions],
            }
            episodes_data.append(ep_data)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rl_trajectories_{timestamp}.json"

        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "num_episodes": len(episodes_data),
                "total_transitions": self.total_transitions,
                "intent_list": INTENT_LIST,
                "episodes": episodes_data,
            }, f, ensure_ascii=False, indent=2)
        print(f"[DataCollector] RL轨迹数据已导出: {filepath}")
        print(f"  Episodes: {len(episodes_data)}, Transitions: {self.total_transitions}")
        return filepath

    def export_csv(self, filename: str = None) -> str:
        """导出为CSV格式（方便分析）"""
        import csv

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.csv"

        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "step", "state_mean", "state_std",
                           "action", "intent", "reward", "done"])
            for ep in self.episodes:
                for i, t in enumerate(ep.transitions):
                    writer.writerow([
                        ep.episode_id, i,
                        float(np.mean(t.state)), float(np.std(t.state)),
                        t.action, INTENT_LIST[t.action],
                        t.reward, t.done,
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

        lines.append("=" * 60)
        return "\n".join(lines)

    def reset(self):
        """重置所有数据"""
        self.episodes.clear()
        self.current_episode = None
        self.total_transitions = 0
        self.action_counts = {i: 0 for i in range(len(INTENT_LIST))}
        self.reward_history.clear()
        self.episode_rewards.clear()
