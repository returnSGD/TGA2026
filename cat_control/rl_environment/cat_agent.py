"""
《猫语心声》 —— 猫咪智能体

整合：状态、记忆、行为树、性格过滤器
每个决策周期执行：意图选择 → 行为树执行 → 状态更新 → 记忆存储
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import random

from .config import (
    INTENT_LIST, PERSONALITY_KEYS, PERSONALITY_DIM,
    EMOTION_DIM, PHYSICAL_DIM, ENV_FEATURE_DIM, RELATION_DIM,
    PLAYER_ACTION_DIM, MEMORY_EMBED_DIM, TOP_K_MEMORIES, SEQ_LEN,
    CAT_CONFIGS, REWARDS, EVENT_IMPORTANCE_MID,
)
from .cat_state import CatState, MemoryItem, MemoryManager
from .bt_core import BTStatus, Blackboard, BehaviorTree
from .bt_intents import build_bt_for_intent, get_all_behavior_trees
from .personality_filter import PersonalityFilter


class CatAgent:
    """
    猫咪智能体 —— 整合所有AI模块

    职责：
    1. 维护猫咪状态
    2. 接收环境信息
    3. 选择意图（规则策略 或 RL策略网络）
    4. 执行行为树
    5. 更新情绪/信任数值
    6. 存储记忆
    """

    def __init__(self, cat_id: str, config: dict = None,
                 personality_filter: PersonalityFilter = None):
        self.cat_id = cat_id
        self.config = config or {}

        # 状态初始化
        self.state = CatState()
        self._init_from_config()

        # 记忆系统
        self.memory_mgr = MemoryManager()

        # 性格过滤器
        self.filter = personality_filter or PersonalityFilter()

        # 行为树缓存
        self._bt_cache: Dict[str, BehaviorTree] = {}
        self._current_bt: Optional[BehaviorTree] = None
        self._bt_in_progress: bool = False

        # 意图历史
        self.intent_history: List[str] = []
        self.state_history: list = []  # 存储完整状态向量

        # 行为参数（由性格过滤器计算）
        self.behavior_params: Dict[str, float] = {}

        # 统计
        self.total_ticks: int = 0
        self.intent_counts: Dict[str, int] = {i: 0 for i in INTENT_LIST}
        self.bt_success_count: int = 0
        self.bt_failure_count: int = 0
        self.interaction_count: int = 0
        self.trust_milestones_reached: List[int] = []

        # 冷却计时器
        self._action_cooldowns: Dict[str, int] = {}

    def _init_from_config(self):
        """从配置字典初始化状态"""
        cfg = self.config
        self.state.personality_vector = np.array(
            cfg.get("personality", [0.0] * PERSONALITY_DIM), dtype=np.float32
        )
        self.state.trust_level = cfg.get("trust_init", 20.0)
        self.state.stress_level = cfg.get("stress_init", 50.0)
        self.state.emotion_vector = np.array([0.3, 0.25, 0.5, 0.5, 0.3], dtype=np.float32)

    @property
    def name(self) -> str:
        return self.config.get("name", self.cat_id)

    @property
    def personality_summary(self) -> str:
        traits = []
        for k, v in zip(PERSONALITY_KEYS, self.state.personality_vector):
            if v > 0.3:
                traits.append(f"{k}({v:.1f})")
        return ", ".join(traits) if traits else "无明显性格"

    # ==================== 决策主循环 ====================

    def decide_intent_with_rule(self, env, epsilon: float = 0.05) -> str:
        """
        使用规则策略（if-else + 随机）选择意图。
        阶段一用于收集训练数据，后续替换为RL策略网络。

        参数:
            env: 沙盒环境
            epsilon: ε-greedy 探索率（0.05 = 5%概率随机探索）
        """
        # ── ε-greedy 探索：以 epsilon 概率随机选择意图（保证数据覆盖） ──
        if random.random() < epsilon:
            # 随机选择但受性格过滤器约束（用性格向量加权）
            return self._exploratory_intent()

        s = self.state

        # 最高优先级：安全/应激行为
        if s.fear > 0.7 or s.stress_level > 80:
            if env.get_nearest_object(s.position, "hiding_box"):
                return "hide"
            return "fearful_retreat"

        if s.fear > 0.5:
            if env.player_action in ("approach", "grab"):
                return "fearful_retreat"
            if random.random() < 0.6:
                return "hide"

        # 生理需求优先级
        if s.hunger > 0.6 and env.get_nearest_object(s.position, "food_bowl"):
            return "eat"
        if s.energy < 0.2 and s.fear < 0.5:
            return "sleep"

        # 玩家互动
        player_dist = env.manhattan_distance(s.position, env.player_position)
        if player_dist <= 5:
            if env.player_action == "pet" and s.trust_level > 30:
                if s.fear < 0.4:
                    return random.choice(["accept_petting", "accept_petting", "approach_player"])
            if env.player_action == "call" and s.trust_level > 50:
                return "approach_player"

        # 社交需求
        nearby = env.get_nearby_cats(self.cat_id, max_dist=3)
        if nearby and s.social_need > 0.5 and s.energy > 0.4:
            if random.random() < 0.4:
                return random.choice(["social_groom", "social_play"])

        # 探索与玩耍
        if s.curiosity > 0.6:
            toys = [o for o in env.objects.values() if o.is_toy and
                    env.manhattan_distance(s.position, o.position) < 5]
            if toys:
                return "play_with_toy"
            return "curious_inspect"

        # 日常行为
        if s.energy < 0.3:
            return "sleep"

        # 信任足够则倾向靠近玩家
        if s.trust_level > 60 and player_dist <= 6:
            if random.random() < 0.4:
                return "approach_player"
            if random.random() < 0.2:
                return "ask_for_attention"

        # 窗边
        if s.comfort > 0.5 and env.get_nearest_object(s.position, "window_spot"):
            if random.random() < 0.2:
                return "stare_at_window"

        return "idle_wander"

    def _exploratory_intent(self) -> str:
        """ε-greedy 探索：基于性格向量加权随机选择意图"""
        # 对每个意图计算性格兼容度作为选择权重
        weights = []
        for intent in INTENT_LIST:
            # 基础权重为1，加上性格矩阵中所有正偏置的加权和
            w = 0.5  # 基础最小值
            for j, trait in enumerate(PERSONALITY_KEYS):
                p = self.state.personality_vector[j]
                if p > 0.01:
                    bias = self.filter._intent_trait_matrix.get(trait, {}).get(intent, 0.0)
                    if bias > 0:
                        w += bias * p
            # 即使是负偏置的意图也保留极小概率
            weights.append(max(0.1, w))

        return random.choices(INTENT_LIST, weights=weights, k=1)[0]

    def process_interaction(self, env, player_action: str = "none",
                           force_intent: str = None) -> Dict:
        """
        处理一个决策周期（1 tick）的主入口。

        参数:
            env: SandboxEnvironment
            player_action: 玩家当前行为
            force_intent: 强制设定意图（用于调试/测试）

        返回: {
            "intent": 选择的意图,
            "bt_status": 行为树状态,
            "action": 当前原子动作,
            "reward": 奖励值（用于RL训练）,
            "memories_retrieved": 检索到的记忆数,
            "state_summary": 状态摘要,
        }
        """
        self.total_ticks += 1

        # 更新环境信息
        self.state.environment_features = np.array([
            env.get_room_env(self.state.current_room_id).get("comfort", 0.5),
            env.get_room_env(self.state.current_room_id).get("stimulation", 0.3),
            env.get_room_env(self.state.current_room_id).get("hygiene", 0.5),
            env.get_room_env(self.state.current_room_id).get("light", 0.5),
            env.get_room_env(self.state.current_room_id).get("noise", 0.3),
        ], dtype=np.float32)

        self.state.relation_vector = self._calc_relations(env)
        self.state.nearby_cats = env.get_nearby_cats(self.cat_id, max_dist=3)

        # 更新行为参数
        self.behavior_params = self.filter.get_behavior_params(
            self.state.personality_vector
        )

        # 需求衰减
        env.apply_need_decay(self.state)

        # 记忆检索（用于状态向量构建）
        # 查询向量需与记忆嵌入同维度（128），用完整状态向量作为查询
        query_vec = self.state.to_state_vector()  # 约42维，零填充到128维
        if len(query_vec) < MEMORY_EMBED_DIM:
            padded = np.zeros(MEMORY_EMBED_DIM, dtype=np.float32)
            padded[:len(query_vec)] = query_vec
            query_vec = padded
        similar_memories = self.memory_mgr.query_similar(query_vec, top_k=TOP_K_MEMORIES)

        # 意图选择
        if force_intent:
            selected_intent = force_intent
        else:
            selected_intent = self.decide_intent_with_rule(env)

        # 意图变化处理
        if selected_intent != self.state.current_intent:
            self.state.last_intent_change_tick = self.total_ticks
            # 检查是否连续同一意图失败
            if self._bt_in_progress and self._current_bt and \
               self._current_bt.last_status == BTStatus.FAILURE:
                self.state.consecutive_intent_failures += 1
            else:
                self.state.consecutive_intent_failures = 0

        self.state.current_intent = selected_intent
        self.intent_history.append(selected_intent)
        self.intent_counts[selected_intent] += 1

        if len(self.intent_history) > 100:
            self.intent_history = self.intent_history[-100:]

        # 构建/获取行为树
        bt = self._get_or_build_bt(selected_intent)

        # 准备黑板
        bt.blackboard.set("state", self.state)
        bt.blackboard.set("env", env)
        bt.blackboard.set("cat_id", self.cat_id)
        bt.blackboard.set("behavior_params", self.behavior_params)
        bt.blackboard.set("hunger_threshold",
                         0.5 + 0.3 * self.state.personality_vector[3])  # 贪吃性格阈值更低
        bt.blackboard.set("interest_threshold", 0.3)
        bt.blackboard.set("_report_msg", None)

        # 执行行为树
        bt_status = bt.tick()
        self._current_bt = bt
        self._bt_in_progress = (bt_status == BTStatus.RUNNING)

        if bt_status == BTStatus.SUCCESS:
            self.bt_success_count += 1
        elif bt_status == BTStatus.FAILURE:
            self.bt_failure_count += 1

        # 更新情绪/信任数值
        reward = self._update_emotional_state(selected_intent, player_action, bt_status)

        # 检查信任里程碑
        self._check_trust_milestone()

        # 存储记忆
        self._store_memory(selected_intent, player_action, bt_status, env)

        # 存储状态历史
        state_vec = self.state.to_state_vector()
        self.state_history.append(state_vec)
        if len(self.state_history) > SEQ_LEN * 10:
            self.state_history = self.state_history[-SEQ_LEN * 5:]

        return {
            "intent": selected_intent,
            "bt_status": bt_status,
            "action": self.state.current_action,
            "reward": reward,
            "memories_retrieved": len(similar_memories),
            "state_summary": self.state.summary(),
            "behavior_params": dict(self.behavior_params),
        }

    # ==================== 内部方法 ====================

    def _get_or_build_bt(self, intent: str) -> BehaviorTree:
        """获取或构建意图对应的行为树"""
        if intent not in self._bt_cache:
            self._bt_cache[intent] = build_bt_for_intent(intent)
        bt = self._bt_cache[intent]
        # 重置行为树（新决策周期）
        if bt.last_status != BTStatus.RUNNING:
            bt.reset()
        return bt

    def _calc_relations(self, env) -> np.ndarray:
        nearby = env.get_nearby_cats(self.cat_id, max_dist=5)
        # 简化计算
        avg_affinity = 0.3
        avg_hostility = 0.1
        social_rank = 0.5

        return np.array([
            self.state.trust_level / 100.0,
            avg_affinity,
            avg_hostility,
            social_rank,
        ], dtype=np.float32)

    def _update_emotional_state(self, intent: str, player_action: str,
                                bt_status: BTStatus) -> float:
        """基于意图、玩家行为和BT结果更新情绪数值，返回奖励"""
        s = self.state
        reward = 0.0

        # 进食完成
        if intent == "eat" and bt_status == BTStatus.SUCCESS:
            s.hunger = max(0.0, s.hunger - 0.4)
            s.comfort += 0.05
            reward = REWARDS["eat_success"]

        # 睡眠完成
        if intent == "sleep" and bt_status == BTStatus.SUCCESS:
            s.energy = min(1.0, s.energy + 0.5)
            s.stress_level -= 10
            reward = REWARDS["sleep_success"]

        # 玩耍完成
        if intent in ("play_with_toy", "social_play") and bt_status == BTStatus.SUCCESS:
            s.curiosity -= 0.15
            s.comfort += 0.08
            s.stress_level -= 8
            if intent == "social_play":
                reward = REWARDS["social_positive"]

        # 社交舔毛完成
        if intent == "social_groom" and bt_status == BTStatus.SUCCESS:
            s.social_need = max(0.0, s.social_need - 0.3)
            reward = REWARDS["social_positive"]

        # 躲藏完成
        if intent == "hide" and bt_status == BTStatus.SUCCESS:
            s.fear = max(0.0, s.fear - 0.15)
            s.stress_level -= 8
            reward = REWARDS["hide_success"]

        # 靠近玩家并被接受
        if intent == "approach_player" and bt_status == BTStatus.SUCCESS:
            if player_action in ("pet", "treat", "call"):
                s.trust_level = min(100, s.trust_level + 2)
                reward = REWARDS["approach_accepted"]

        # 抚摸互动
        if intent == "accept_petting" and bt_status == BTStatus.SUCCESS:
            s.trust_level = min(100, s.trust_level + 3)
            s.comfort += 0.1
            s.stress_level -= 5
            reward = REWARDS["player_interact"]

        # 恐惧撤退
        if intent == "fearful_retreat" and bt_status == BTStatus.SUCCESS:
            s.fear = max(0.0, s.fear - 0.05)
            if player_action != "grab":
                s.trust_level += 1  # 玩家没有追逐，微增信任
                reward = 0.2

        # 玩家交互通用影响
        if player_action == "pet" and intent == "eat":
            s.stress_level -= 2  # 进食时被抚摸会降低压力
        if player_action == "scold" and intent in ("play_with_toy", "idle_wander"):
            s.stress_level += 5
            reward -= 0.2
        if player_action == "ignore":
            # 长期忽视有代价
            if s.social_need > 0.8:
                s.stress_level += 1

        # 意图失败的影响
        if bt_status == BTStatus.FAILURE:
            s.stress_level += 2
            reward -= 0.1

        # 情绪数值边界
        s.hunger = np.clip(s.hunger, 0.0, 1.0)
        s.fear = np.clip(s.fear, 0.0, 1.0)
        s.curiosity = np.clip(s.curiosity, 0.0, 1.0)
        s.comfort = np.clip(s.comfort, 0.0, 1.0)
        s.social_need = np.clip(s.social_need, 0.0, 1.0)
        s.stress_level = np.clip(s.stress_level, 0.0, 100.0)
        s.trust_level = np.clip(s.trust_level, 0.0, 100.0)

        return reward

    def _check_trust_milestone(self):
        """检查是否触发信任里程碑"""
        milestones = {20: "初步探索", 40: "试探性接触", 65: "深度信赖", 90: "彻底安心"}
        for threshold, name in milestones.items():
            if (self.state.trust_level >= threshold and
                threshold not in self.trust_milestones_reached):
                self.trust_milestones_reached.append(threshold)
                if hasattr(self, '_env_ref'):
                    self._env_ref.log_event(
                        f"🌟 {self.name} 触发信任里程碑: {name} (信任 {threshold}%)"
                    )

    def _store_memory(self, intent: str, player_action: str,
                      bt_status: BTStatus, env):
        """存储交互记忆（使用 EVENT_IMPORTANCE_MID 计算重要性）"""
        success = bt_status == BTStatus.SUCCESS

        # 根据事件类型从配置中获取基础重要性
        event_type = self._classify_event(intent)
        base_importance = EVENT_IMPORTANCE_MID.get(
            event_type,
            EVENT_IMPORTANCE_MID.get("daily_feed", 4.0)
        )

        # 根据意图和玩家行为微调
        if success:
            base_importance = min(10.0, base_importance * 1.2)
        else:
            base_importance = max(1.0, base_importance * 0.8)

        # 高信任度时记忆稍有增值
        if self.state.trust_level > 60:
            base_importance += 0.5 * (self.state.trust_level / 100)

        # 小幅随机抖动防止所有记忆分值完全相同
        importance = base_importance + random.uniform(-0.5, 0.5)

        desc = f"{'成功' if success else '失败'}执行{intent}"
        if player_action != "none":
            desc += f"，玩家{player_action}"

        mem = MemoryItem(
            desc=desc,
            timestamp=env.game_tick,
            importance=importance,
            embedding=np.random.randn(MEMORY_EMBED_DIM).astype(np.float32),
            event_type=event_type,
        )
        self.memory_mgr.add_memory(mem)
        self.interaction_count += 1

    def _classify_event(self, intent: str) -> str:
        """分类事件类型（映射到 EVENT_IMPORTANCE_BASE 的键）"""
        if intent in ("fearful_retreat", "hiss_warning"):
            return "trauma_triggered" if self.state.fear > 0.7 else "scared_by_player"
        if intent == "hide":
            return "trauma_triggered" if self.state.stress_level > 80 else "scared_by_player"
        if intent == "approach_player":
            return "first_voluntary_rub" if self.state.trust_level > 60 else "routine_explore"
        if intent == "accept_petting":
            return "first_pet_accepted" if len(self.trust_milestones_reached) <= 1 else "routine_pet_accepted"
        if intent == "ask_for_attention":
            return "first_voluntary_rub"
        if intent == "eat":
            return "daily_feed"
        if intent == "sleep":
            return "routine_sleep"
        if intent in ("social_groom", "social_play"):
            return "first_grooming_together" if self.state.social_need > 0.6 else "social_bond_formed"
        if intent in ("curious_inspect", "play_with_toy"):
            return "routine_explore" if self.state.curiosity < 0.5 else "player_soothe_success"
        if intent == "follow_player":
            return "first_night_sleep_near_player" if self.state.trust_level > 70 else "routine_explore"
        if intent == "stare_at_window":
            return "stare_window"
        return "idle_wander"

    # ==================== 数据收集相关 ====================

    def build_full_state(self, player_action: str,
                         memory_embeds: List[np.ndarray] = None) -> np.ndarray:
        """构建完整RL输入状态向量"""
        action_list = ["pet", "feed", "call", "play", "ignore", "scold",
                       "approach", "leave", "treat", "heal", "photo", "none"]
        action_embed = np.zeros(PLAYER_ACTION_DIM, dtype=np.float32)
        if player_action in action_list:
            action_embed[action_list.index(player_action)] = 1.0

        memory_embeds = memory_embeds or []
        memory_parts = []
        for i in range(TOP_K_MEMORIES):
            if i < len(memory_embeds):
                memory_parts.append(memory_embeds[i])
            else:
                memory_parts.append(np.zeros(MEMORY_EMBED_DIM, dtype=np.float32))

        return np.concatenate([
            self.state.to_state_vector(),
            action_embed,
            *memory_parts
        ]).astype(np.float32)

    def get_state_sequence(self) -> np.ndarray:
        """构造序列输入（当前+前3步）"""
        seq = list(self.state_history)
        while len(seq) < SEQ_LEN:
            if seq:
                seq.insert(0, np.zeros_like(seq[0]))
            else:
                seq.insert(0, np.zeros(422, dtype=np.float32))
        return np.stack(seq[-SEQ_LEN:], axis=0)

    # ==================== 统计与调试 ====================

    def stats_summary(self) -> str:
        total = sum(self.intent_counts.values()) or 1
        top_intents = sorted(self.intent_counts.items(), key=lambda x: -x[1])[:5]
        intent_str = ", ".join(f"{i}={c}({c/total:.0%})" for i, c in top_intents)

        return (
            f"{self.name} | 信任:{self.state.trust_level:.0f} 压力:{self.state.stress_level:.0f} | "
            f"BT成功:{self.bt_success_count} 失败:{self.bt_failure_count} "
            f"({self.bt_success_count/max(1,self.bt_success_count+self.bt_failure_count):.0%}) | "
            f"交互:{self.interaction_count} | "
            f"Top意图:[{intent_str}]"
        )
