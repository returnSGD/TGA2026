"""
《猫语心声》 —— 规则策略（阶段一数据收集用）

if-else + 随机 的规则策略，模拟猫咪行为决策。
用于沙盒中生成 (state, action) 训练对，为后续RL行为克隆预热提供数据。
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import random
import numpy as np

from .config import INTENT_LIST


class RuleStrategy:
    """
    规则策略 —— 决策优先级：

    1. 应激层：恐惧/压力超高 → hide/fearful_retreat
    2. 生理层：饥饿/疲劳 → eat/sleep
    3. 交互层：玩家在附近 → 根据信任度选择互动
    4. 社交层：附近有猫 → social_groom/social_play
    5. 探索层：好奇高 → curious_inspect/play_with_toy
    6. 日常层：idle_wander/stare_at_window/sleep
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def select_intent(self, state, env, cat_id: str = "") -> str:
        """
        根据规则选择意图。

        参数:
            state: CatState 对象
            env: SandboxEnvironment 对象
            cat_id: 猫咪ID（可选，用于社交层判断附近猫）
        返回: 意图名字符串
        """
        s = state

        # ---- Layer 1: 应激行为 ----
        if s.fear > 0.8 or s.stress_level > 85:
            if env.get_nearest_object(s.position, "hiding_box"):
                return "hide"
            return "fearful_retreat"

        if s.fear > 0.6:
            if env.player_action in ("approach", "grab", "scold"):
                return "fearful_retreat"
            if self.rng.random() < 0.7:
                return "hide"

        # ---- Layer 2: 生理需求 ----
        if s.hunger > 0.65 and env.get_nearest_object(s.position, "food_bowl"):
            return "eat"

        if s.energy < 0.15:
            return "sleep"
        if s.energy < 0.3 and s.fear < 0.5 and self.rng.random() < 0.6:
            return "sleep"

        # ---- Layer 3: 玩家互动 ----
        player_obj = env.get_nearest_object(s.position, "player") or \
                     type('obj', (), {'position': env.player_position})()
        player_dist = env.manhattan_distance(s.position, env.player_position)

        if player_dist <= 5:
            action = env.player_action

            if action == "pet" and s.trust_level > 30 and s.fear < 0.5:
                return self.rng.choice([
                    "accept_petting", "accept_petting",
                    "accept_petting", "approach_player"
                ])

            if action == "call" and s.trust_level > 50 and s.fear < 0.4:
                return self.rng.choice([
                    "approach_player", "approach_player", "follow_player"
                ])

            if action == "treat" and s.hunger > 0.3:
                return "eat" if env.get_nearest_object(s.position, "food_bowl") \
                       else "approach_player"

            if action == "play" and s.energy > 0.4:
                return self.rng.choice(["play_with_toy", "approach_player"])

            # 信任够高主动互动
            if s.trust_level > 70 and player_dist <= 3 and s.fear < 0.3:
                if self.rng.random() < 0.25:
                    return self.rng.choice(["ask_for_attention", "approach_player"])

            # 高社交需求 + 玩家在附近
            if s.social_need > 0.7 and s.trust_level > 50 and player_dist <= 4:
                if self.rng.random() < 0.2:
                    return "approach_player"

        # ---- Layer 4: 社交行为 ----
        nearby = env.get_nearby_cats(cat_id, max_dist=4) if cat_id else []
        if s.social_need > 0.5 and s.energy > 0.35 and s.fear < 0.5:
            if self.rng.random() < 0.3:
                return self.rng.choice(["social_groom", "social_play"])

        # ---- Layer 5: 探索与玩耍 ----
        if s.curiosity > 0.65:
            toys = [o for o in env.objects.values() if o.is_toy]
            if toys:
                nearest_toy = min(toys, key=lambda o:
                    env.manhattan_distance(s.position, o.position))
                if env.manhattan_distance(s.position, nearest_toy.position) < 6:
                    return "play_with_toy"
            return "curious_inspect"

        if s.curiosity > 0.45 and self.rng.random() < 0.3:
            if self.rng.random() < 0.5:
                return "curious_inspect"
            return "stare_at_window"

        # ---- Layer 6: 日常默认 ----
        if s.energy < 0.35 and s.fear < 0.5:
            return "sleep"

        if s.comfort > 0.6 and self.rng.random() < 0.2:
            return "stare_at_window"

        return "idle_wander"

    def select_random_intent(self, state=None, env=None) -> str:
        """完全随机选择意图（用于探索数据收集）"""
        return self.rng.choice(INTENT_LIST)

    def select_weighted_intent(self, state, env, weights: dict = None) -> str:
        """按权重随机选择意图（适合性权重调优）"""
        intents = INTENT_LIST
        w = [weights.get(i, 1.0) for i in intents] if weights else [1.0] * len(intents)
        return self.rng.choices(intents, weights=w, k=1)[0]
