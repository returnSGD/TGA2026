"""
《猫语心声》 —— Gym-like环境包装器

将 SandboxEnvironment 包装为标准的 reset()/step(action) 接口，
适配PPO训练框架。支持单猫模式和多猫自对弈模式。
"""

from __future__ import annotations
import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_environment.config import (
    INTENT_LIST, PERSONALITY_DIM, PLAYER_ACTION_DIM,
    MEMORY_EMBED_DIM, TOP_K_MEMORIES, STATE_DIM, REWARDS,
    CAT_CONFIGS,
)
from rl_environment.environment import SandboxEnvironment
from rl_environment.cat_agent import CatAgent
from rl_environment.personality_filter import PersonalityFilter
from rl_environment.bt_core import BTStatus


class SingleCatEnv:
    """
    单猫咪PPO训练环境。

    每 step 推进一个 tick：猫咪选择意图 → 行为树执行 → 返回 (obs, reward, done, info)
    """

    def __init__(self, cat_id: str = "oreo",
                 max_steps_per_episode: int = 2048,
                 seed: int = 42):
        self.cat_id = cat_id
        self.max_steps = max_steps_per_episode
        self.seed = seed

        # 环境
        self.env = SandboxEnvironment(seed=seed)
        self.pf = PersonalityFilter()
        self.cat: Optional[CatAgent] = None
        self.current_step = 0
        self.episode_reward = 0.0

        # 记忆嵌入缓存（上次检索结果，用于构建当前状态）
        self._last_memory_embeds: List[np.ndarray] = []

        # 玩家行为序列（模拟）
        self._player_actions = deque([
            "none", "none", "pet", "none", "none", "feed", "none",
            "call", "none", "none", "treat", "none", "play", "none",
            "none", "none", "pet", "none", "approach", "none",
        ], maxlen=50)

    def reset(self, seed: int = None) -> np.ndarray:
        """重置环境，返回初始观测"""
        if seed is not None:
            self.seed = seed
        self.env = SandboxEnvironment(seed=self.seed)
        self.pf = PersonalityFilter()

        cfg = CAT_CONFIGS.get(self.cat_id, CAT_CONFIGS["oreo"])
        self.cat = CatAgent(cat_id=self.cat_id, config=cfg,
                           personality_filter=self.pf)

        # 初始化位置
        init_positions = {"xiaoxue": (2, 3), "oreo": (6, 4), "orange": (10, 5)}
        pos = init_positions.get(self.cat_id, (6, 4))
        self.cat.state.position = pos
        self.cat.state.current_room_id = self.env.get_room_id_at(pos)
        self.env.cat_positions[self.cat_id] = pos

        self.current_step = 0
        self.episode_reward = 0.0
        self._last_memory_embeds = []

        return self._get_obs()

    def step(self, intent_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步。
        参数:
            intent_idx: 0~14 的意图索引
        返回:
            obs, reward, terminated, truncated, info
        """
        intent = INTENT_LIST[intent_idx]

        # 推进环境tick
        self.env.advance_tick()

        # 生成玩家行为
        if self.current_step % 3 == 0:
            self._player_actions.rotate(-1)
        player_action = self._player_actions[0]
        self.env.set_player_action(player_action)

        # 更新猫的环境信息
        self.cat.state.environment_features = np.array([
            self.env.get_room_env(self.cat.state.current_room_id).get("comfort", 0.5),
            self.env.get_room_env(self.cat.state.current_room_id).get("stimulation", 0.3),
            self.env.get_room_env(self.cat.state.current_room_id).get("hygiene", 0.5),
            self.env.get_room_env(self.cat.state.current_room_id).get("light", 0.5),
            self.env.get_room_env(self.cat.state.current_room_id).get("noise", 0.3),
        ], dtype=np.float32)
        self.env.apply_need_decay(self.cat.state)

        # 执行交互（强制使用RL选择的意图）
        result = self.cat.process_interaction(
            self.env, player_action=player_action, force_intent=intent
        )

        reward = result["reward"]
        self.episode_reward += reward
        self.current_step += 1

        # 判断终止条件
        terminated = False
        truncated = self.current_step >= self.max_steps
        if self.cat.state.health < 0.1:
            terminated = True
        if self.cat.state.trust_level >= 95:
            terminated = True  # 充分信任，episode成功完成

        obs = self._get_obs()
        info = {
            "intent": intent,
            "bt_status": result["bt_status"],
            "reward": reward,
            "episode_reward": self.episode_reward,
            "trust": self.cat.state.trust_level,
            "stress": self.cat.state.stress_level,
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """构建422维状态观测"""
        if self.cat is None:
            return np.zeros(STATE_DIM, dtype=np.float32)

        player_action = self._player_actions[0] if self._player_actions else "none"
        state_vec = self.cat.state.to_state_vector()
        query_vec = np.zeros(MEMORY_EMBED_DIM, dtype=np.float32)
        query_vec[:len(state_vec)] = state_vec
        memory_embeds = self.cat.memory_mgr.get_memory_embeddings(query_vec)
        self._last_memory_embeds = memory_embeds

        return self.cat.build_full_state(player_action, memory_embeds)

    def get_personality(self) -> np.ndarray:
        """返回当前猫的性格向量 [8]"""
        if self.cat is None:
            return np.zeros(PERSONALITY_DIM, dtype=np.float32)
        return self.cat.state.personality_vector.copy()


class MultiCatEnv:
    """
    多猫咪自对弈PPO训练环境。

    同时管理3只猫，每只猫独立决策，共享环境。
    支持共享策略网络（cated_share_policy=True），每只猫通过不同的性格嵌入条件化。
    """

    def __init__(self, cat_ids: List[str] = None,
                 max_steps_per_episode: int = 2048,
                 seed: int = 42):
        self.cat_ids = cat_ids or list(CAT_CONFIGS.keys())[:3]
        self.num_cats = len(self.cat_ids)
        self.max_steps = max_steps_per_episode
        self.seed = seed

        # 共享环境
        self.env = SandboxEnvironment(seed=seed)
        self.pf = PersonalityFilter()
        self.cats: Dict[str, CatAgent] = {}
        self.current_step = 0
        self.episode_rewards: Dict[str, float] = {cid: 0.0 for cid in self.cat_ids}

        # 玩家行为
        self._player_actions = deque([
            "none", "none", "pet", "none", "feed", "none", "call",
            "none", "treat", "none", "play", "none", "none", "pet",
        ], maxlen=50)

    def reset(self, seed: int = None) -> Dict[str, np.ndarray]:
        """重置环境，返回所有猫的初始观测"""
        if seed is not None:
            self.seed = seed
        self.env = SandboxEnvironment(seed=self.seed)
        self.pf = PersonalityFilter()
        self.cats.clear()

        positions = {"xiaoxue": (2, 3), "oreo": (6, 4), "orange": (10, 5)}
        for cid in self.cat_ids:
            cfg = CAT_CONFIGS.get(cid, {"personality": [0.0]*8, "trust_init": 30})
            cat = CatAgent(cat_id=cid, config=cfg, personality_filter=self.pf)
            pos = positions.get(cid, (6, 4))
            cat.state.position = pos
            cat.state.current_room_id = self.env.get_room_id_at(pos)
            self.env.cat_positions[cid] = pos
            self.cats[cid] = cat

        self.current_step = 0
        self.episode_rewards = {cid: 0.0 for cid in self.cat_ids}

        return {cid: self._get_obs(cid) for cid in self.cat_ids}

    def step(self, intent_dict: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, float],
        Dict[str, bool], Dict[str, bool], Dict[str, Dict]
    ]:
        """
        执行一步（所有猫同时决策）。

        参数:
            intent_dict: {cat_id: intent_idx} 每只猫选择的意图

        返回:
            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict
        """
        self.env.advance_tick()
        if self.current_step % 3 == 0:
            self._player_actions.rotate(-1)
        player_action = self._player_actions[0]
        self.env.set_player_action(player_action)

        obs_dict = {}
        reward_dict = {}
        terminated_dict = {}
        truncated_dict = {}
        info_dict = {}

        for cid in self.cat_ids:
            cat = self.cats[cid]
            intent_idx = intent_dict.get(cid, 0)
            intent = INTENT_LIST[intent_idx]

            cat.state.environment_features = np.array([
                self.env.get_room_env(cat.state.current_room_id).get("comfort", 0.5),
                self.env.get_room_env(cat.state.current_room_id).get("stimulation", 0.3),
                self.env.get_room_env(cat.state.current_room_id).get("hygiene", 0.5),
                self.env.get_room_env(cat.state.current_room_id).get("light", 0.5),
                self.env.get_room_env(cat.state.current_room_id).get("noise", 0.3),
            ], dtype=np.float32)
            self.env.apply_need_decay(cat.state)

            result = cat.process_interaction(
                self.env, player_action=player_action, force_intent=intent
            )

            reward = result["reward"]
            self.episode_rewards[cid] += reward

            obs_dict[cid] = self._get_obs(cid)
            reward_dict[cid] = reward
            terminated_dict[cid] = (cat.state.health < 0.1 or cat.state.trust_level >= 95)
            truncated_dict[cid] = self.current_step >= self.max_steps
            info_dict[cid] = {
                "intent": intent,
                "bt_status": result["bt_status"],
                "reward": reward,
                "episode_reward": self.episode_rewards[cid],
                "trust": cat.state.trust_level,
                "stress": cat.state.stress_level,
            }

        self.current_step += 1
        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    def _get_obs(self, cat_id: str) -> np.ndarray:
        cat = self.cats.get(cat_id)
        if cat is None:
            return np.zeros(STATE_DIM, dtype=np.float32)
        player_action = self._player_actions[0] if self._player_actions else "none"
        state_vec = cat.state.to_state_vector()
        query_vec = np.zeros(MEMORY_EMBED_DIM, dtype=np.float32)
        query_vec[:len(state_vec)] = state_vec
        memory_embeds = cat.memory_mgr.get_memory_embeddings(query_vec)
        return cat.build_full_state(player_action, memory_embeds)

    def get_personalities(self) -> Dict[str, np.ndarray]:
        return {cid: cat.state.personality_vector.copy()
                for cid, cat in self.cats.items()}

    def any_terminated(self, d: Dict[str, bool]) -> bool:
        return any(d.values())

    def all_terminated(self, d: Dict[str, bool]) -> bool:
        return all(d.values())
