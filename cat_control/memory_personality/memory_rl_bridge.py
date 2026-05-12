"""
《猫语心声》 —— 记忆→RL状态注入桥梁

技术策划案v2 §4.3.3 的完整实现：
- 从当前状态(情绪+环境)构建查询向量
- 检索Top-3记忆嵌入并注入RL状态向量
- 时间衰减调度 + 记忆压缩触发
- 经验→记忆自动记录
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional

from .config import MemoryConfig, QUERY_VECTOR_DIM
from .memory_manager import MemoryManager
from .personality_filter import PersonalityFilter
from rl_environment.config import (
    EMOTION_DIM, ENV_FEATURE_DIM, MEMORY_EMBED_DIM,
    TOP_K_MEMORIES, STATE_DIM, PERSONALITY_DIM,
    PLAYER_ACTION_DIM, INTENT_LIST,
)


class MemoryRLBridge:
    """
    记忆系统与RL策略网络的桥接层。

    职责：
    1. 查询向量构建：从(情绪, 环境) → 10维查询向量
    2. 记忆注入：检索Top-3 → 拼接到状态向量末尾(位置38~421)
    3. 衰减调度：每N tick触发时间衰减
    4. 压缩调度：每天结束时检查是否需要压缩旧记忆
    5. 经验记录：RL交互结果自动记录为记忆
    """

    def __init__(self, memory_manager: MemoryManager = None,
                 personality_filter: PersonalityFilter = None,
                 config: MemoryConfig = None):
        self.cfg = config or MemoryConfig()
        self.memory = memory_manager or MemoryManager(config=self.cfg)
        self.pf = personality_filter or PersonalityFilter(config=self.cfg)

        # 调度计数
        self._tick_counter = 0
        self._decay_interval = 144        # 每1游戏天衰减一次
        self._compress_interval = 144 * 7  # 每7游戏天检查压缩

        # 状态向量各段的偏移量
        self._personality_offset = 0
        self._emotion_offset = PERSONALITY_DIM                     # 8
        self._physical_offset = self._emotion_offset + EMOTION_DIM  # 13
        self._trust_offset = self._physical_offset + 3              # 16
        self._env_offset = self._trust_offset + 1                   # 17
        self._relation_offset = self._env_offset + ENV_FEATURE_DIM  # 22
        self._action_offset = self._relation_offset + 4             # 26
        self._memory_offset = self._action_offset + PLAYER_ACTION_DIM  # 38

        # 统计
        self.total_injections = 0

    # ═══════════════════════════════════════════
    #  查询向量构建
    # ═══════════════════════════════════════════

    def build_query_vector(self, emotion_vec: np.ndarray,
                           env_vec: np.ndarray) -> np.ndarray:
        """
        从当前情绪和环境特征构建记忆查询向量。

        参数:
            emotion_vec: [5] 情绪向量 (饥饿,恐惧,好奇,舒适,社交需求)
            env_vec: [5] 环境特征 (舒适度,刺激度,卫生度,光照,噪音)

        返回: [128] 查询向量（填充到嵌入维度）
        """
        emotion_weight = self.cfg.query_emotion_weight
        env_weight = self.cfg.query_env_weight

        query_10d = np.concatenate([
            np.asarray(emotion_vec, dtype=np.float32) * emotion_weight,
            np.asarray(env_vec, dtype=np.float32) * env_weight,
        ])

        # 填充到128维
        full_query = np.zeros(MEMORY_EMBED_DIM, dtype=np.float32)
        full_query[:QUERY_VECTOR_DIM] = query_10d
        return full_query

    def build_query_from_state(self, state_vector: np.ndarray) -> np.ndarray:
        """
        从完整RL状态向量中提取情绪+环境部分构建查询向量。

        state_vector: [422] 完整RL状态
        返回: [128] 查询向量
        """
        emotion = state_vector[self._emotion_offset:
                               self._emotion_offset + EMOTION_DIM]
        env = state_vector[self._env_offset:
                           self._env_offset + ENV_FEATURE_DIM]
        return self.build_query_vector(emotion, env)

    # ═══════════════════════════════════════════
    #  记忆注入
    # ═══════════════════════════════════════════

    def inject_memories(self, state_vector: np.ndarray,
                        query_vector: np.ndarray = None) -> np.ndarray:
        """
        将检索到的记忆嵌入注入状态向量。

        state_vector: [422] 或 [batch, 422]，记忆槽位可为零
        query_vector: [128]，为None时从state_vector自动构建

        返回: 注入后的状态向量（新副本）
        """
        state = np.asarray(state_vector, dtype=np.float32).copy()

        if query_vector is None:
            query_vector = self.build_query_from_state(state)

        # 检索Top-K记忆嵌入
        memory_embeds = self.memory.get_memory_embeddings(
            query_vector, top_k=TOP_K_MEMORIES
        )

        # 写入状态向量的记忆段
        for i, emb in enumerate(memory_embeds):
            start = self._memory_offset + i * MEMORY_EMBED_DIM
            end = start + MEMORY_EMBED_DIM
            state[start:end] = emb

        self.total_injections += 1
        return state

    def inject_memories_batch(self, state_vectors: np.ndarray,
                              query_vectors: np.ndarray = None
                              ) -> np.ndarray:
        """
        批量记忆注入（用于RL训练中的并行推理）。

        state_vectors: [batch, 422]
        query_vectors: [batch, 128]，为None时自动构建
        """
        batch = state_vectors.copy()
        for i in range(len(batch)):
            qv = query_vectors[i] if query_vectors is not None else None
            batch[i] = self.inject_memories(batch[i], qv)
        return batch

    def get_memory_slot(self, state_vector: np.ndarray,
                        slot_idx: int = 0) -> np.ndarray:
        """提取状态向量中第slot_idx个记忆槽位"""
        start = self._memory_offset + slot_idx * MEMORY_EMBED_DIM
        return state_vector[start:start + MEMORY_EMBED_DIM].copy()

    # ═══════════════════════════════════════════
    #  衰减与压缩调度
    # ═══════════════════════════════════════════

    def on_tick(self, current_tick: float, force_decay: bool = False,
                force_compress: bool = False):
        """
        每个游戏tick调用，管理记忆维护调度。

        当前策略:
        - 每144 tick（1游戏天）执行时间衰减
        - 每1008 tick（7游戏天）检查记忆压缩
        """
        self._tick_counter += 1

        if force_decay or self._tick_counter % self._decay_interval == 0:
            self.memory.apply_time_decay(current_tick)

        if force_compress or self._tick_counter % self._compress_interval == 0:
            compressed = self.memory.compress_old_memories(current_tick)
            if compressed > 0:
                print(f"[MemoryBridge] tick {current_tick}: "
                      f"压缩 {compressed} 条旧记忆")

    # ═══════════════════════════════════════════
    #  经验→记忆自动记录
    # ═══════════════════════════════════════════

    def record_experience(self, cat_name: str, intent: str,
                          bt_success: bool, reward: float,
                          trust_delta: float, stress_delta: float,
                          player_action: str, timestamp: float,
                          extra_context: str = "") -> Optional[str]:
        """
        将RL交互经验自动记录为记忆。

        仅在以下情况记录（避免噪声）:
        - 高奖励事件 (|reward| > 0.5)
        - 信任显著变化 (|Δtrust| > 1.0)
        - 压力显著变化 (|Δstress| > 5.0)
        - 行为树失败（记录失败原因）

        返回: memory_id 或 None（未记录）
        """
        should_record = (
            abs(reward) > 0.5 or
            abs(trust_delta) > 1.0 or
            abs(stress_delta) > 5.0 or
            not bt_success
        )
        if not should_record:
            return None

        # 推断事件类型
        if intent == "eat":
            event_type = "daily_feed"
        elif intent in ("approach_player", "ask_for_attention",
                        "accept_petting", "follow_player"):
            event_type = "routine_pet_accepted" if bt_success else "scared_by_player"
        elif intent in ("social_groom", "social_play"):
            event_type = "first_grooming_together" if bt_success else "fight_with_other_cat"
        elif intent in ("hide", "fearful_retreat", "hiss_warning"):
            event_type = "trauma_triggered"
        elif intent in ("play_with_toy", "curious_inspect"):
            event_type = "routine_explore"
        elif intent == "sleep":
            event_type = "routine_sleep"
        elif intent == "stare_at_window":
            event_type = "stare_window"
        else:
            event_type = "idle_wander"

        # 计算重要性（基于奖励幅度和信任变化）
        base_importance = 4.0
        base_importance += min(3.0, abs(reward) * 1.5)
        base_importance += min(2.0, abs(trust_delta) * 0.5)
        base_importance = min(10.0, base_importance)

        # 构建描述
        desc_parts = []
        if bt_success:
            desc_parts.append(f"{cat_name}成功执行了{intent}")
        else:
            desc_parts.append(f"{cat_name}尝试{intent}但失败了")
        if abs(reward) > 0.1:
            desc_parts.append(f"获得反馈{reward:+.1f}")
        if abs(trust_delta) > 0.5:
            desc_parts.append(f"信任变化{trust_delta:+.1f}")
        if extra_context:
            desc_parts.append(extra_context)

        desc = "；".join(desc_parts)

        item = self.memory.add_memory(
            desc=desc,
            event_type=event_type,
            timestamp=timestamp,
            importance=base_importance,
            metadata={
                "cat": cat_name,
                "intent": intent,
                "reward": float(reward),
                "trust_delta": float(trust_delta),
                "stress_delta": float(stress_delta),
                "player_action": player_action,
                "bt_success": bt_success,
            },
        )
        return item.memory_id

    def record_milestone(self, cat_name: str, milestone_type: str,
                         desc: str, timestamp: float,
                         importance: float = 9.0) -> str:
        """记录里程碑事件（信任突破、首次靠近等）"""
        item = self.memory.add_memory(
            desc=desc,
            event_type=f"milestone_{milestone_type}",
            timestamp=timestamp,
            importance=importance,
            metadata={
                "cat": cat_name,
                "milestone": milestone_type,
            },
        )
        return item.memory_id

    # ═══════════════════════════════════════════
    #  性格过滤集成
    # ═══════════════════════════════════════════

    def filter_intent_logits(self, logits: np.ndarray,
                             personality_vec: np.ndarray) -> np.ndarray:
        """代理：性格过滤器第一层"""
        return self.pf.filter_intent_logits(logits, personality_vec)

    def filter_batch_logits(self, logits: np.ndarray,
                            personality_vecs: np.ndarray) -> np.ndarray:
        """代理：批量logits过滤"""
        return self.pf.filter_batch_logits(logits, personality_vecs)

    def filter_probs(self, probs: np.ndarray, personality_vec: np.ndarray,
                     temperature: float = 1.0) -> np.ndarray:
        """代理：概率过滤"""
        return self.pf.filter_probs(probs, personality_vec, temperature)

    def get_behavior_params(self, personality_vec: np.ndarray) -> Dict[str, float]:
        """代理：行为参数"""
        return self.pf.get_behavior_params(personality_vec)

    # ═══════════════════════════════════════════
    #  状态提取工具
    # ═══════════════════════════════════════════

    def extract_emotion(self, state_vector: np.ndarray) -> np.ndarray:
        """从状态向量提取情绪段"""
        return state_vector[self._emotion_offset:
                            self._emotion_offset + EMOTION_DIM].copy()

    def extract_env(self, state_vector: np.ndarray) -> np.ndarray:
        """从状态向量提取环境段"""
        return state_vector[self._env_offset:
                            self._env_offset + ENV_FEATURE_DIM].copy()

    def extract_personality(self, state_vector: np.ndarray) -> np.ndarray:
        """从状态向量提取性格段"""
        return state_vector[self._personality_offset:
                            self._personality_offset + PERSONALITY_DIM].copy()

    @staticmethod
    def get_memory_indices() -> Tuple[int, int]:
        """返回记忆段在状态向量中的起止索引"""
        offset = PERSONALITY_DIM + EMOTION_DIM + 3 + 1 + ENV_FEATURE_DIM + 4 + PLAYER_ACTION_DIM
        return offset, offset + MEMORY_EMBED_DIM * TOP_K_MEMORIES

    # ═══════════════════════════════════════════
    #  统计与导出
    # ═══════════════════════════════════════════

    def summary(self) -> str:
        return (
            f"MemoryRLBridge: 注入次数={self.total_injections}, "
            f"tick={self._tick_counter}, "
            f"{self.memory.summary()}"
        )

    def export_state(self) -> Dict:
        return {
            "total_injections": self.total_injections,
            "tick_counter": self._tick_counter,
            "memory_system": self.memory.export_to_dict(),
        }
