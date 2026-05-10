"""
《猫语心声》 —— 性格参数过滤器（三层约束）

第一层：RL意图logits修正
第二层：行为参数调整
第三层：文本关键词过滤（后续阶段实现）
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from .config import (
    INTENT_LIST, PERSONALITY_KEYS, PERSONALITY_DIM,
    INTENT_PERSONALITY_MATRIX, PERSONALITY_BEHAVIOR_PARAMS,
)


class PersonalityFilter:
    """
    三层性格过滤器 —— 保证人设不偏移的最后防线

    第一层：对RL输出的意图logits施加性格偏置
    第二层：根据性格向量计算行为参数
    第三层：LLM文本关键词/情感极性检查（后续实现）
    """

    def __init__(self):
        self._intent_trait_matrix: Dict[str, Dict[str, float]] = INTENT_PERSONALITY_MATRIX
        self._behavior_param_table: Dict[str, Dict[str, float]] = PERSONALITY_BEHAVIOR_PARAMS

        # 默认行为参数（当某性格维度未配置时使用）
        self._default_params = {
            "approach_distance": 1.5,
            "move_speed": 0.6,
            "response_delay": 2.0,
            "flee_distance": 2.0,
            "hesitation_weight": 0.5,
        }

    # ==================== 第一层：意图过滤 ====================

    def filter_intent_logits(self, logits: np.ndarray,
                             personality_vec: np.ndarray) -> np.ndarray:
        """
        对意图概率分布施加性格偏置。
        logits: [num_intents] 维数组
        personality_vec: [8] 维性格向量
        返回: 修正后的logits
        """
        bias = np.zeros_like(logits)

        for i, intent in enumerate(INTENT_LIST):
            for j, trait in enumerate(PERSONALITY_KEYS):
                weight = self._intent_trait_matrix.get(trait, {}).get(intent, 0.0)
                bias[i] += weight * personality_vec[j]

        return logits + bias

    def filter_intent_probs(self, probs: np.ndarray,
                            personality_vec: np.ndarray,
                            temperature: float = 0.1) -> np.ndarray:
        """
        使用softmax + logits偏置方式过滤意图概率分布。
        返回修改后的概率分布（仍归一化）。
        """
        # 防止log(0)
        probs = np.clip(probs, 1e-8, 1.0)
        logits = np.log(probs)
        filtered_logits = self.filter_intent_logits(logits, personality_vec)
        # softmax with temperature
        scaled = filtered_logits / temperature
        scaled -= scaled.max()
        exp = np.exp(scaled)
        return exp / exp.sum()

    # ==================== 第二层：行为参数 ====================

    def get_behavior_params(self, personality_vec: np.ndarray) -> Dict[str, float]:
        """
        根据性格向量计算行为参数。
        返回值：{approach_distance, move_speed, response_delay, flee_distance, hesitation_weight}
        """
        params = {}
        param_names = self._default_params.keys()

        for param_name in param_names:
            value = 0.0
            total_weight = 0.0
            for j, trait in enumerate(PERSONALITY_KEYS):
                p = personality_vec[j]
                if p > 0.01:
                    trait_val = self._behavior_param_table.get(trait, {}).get(param_name)
                    if trait_val is not None:
                        value += trait_val * p
                        total_weight += p

            if total_weight > 0:
                params[param_name] = value / total_weight
            else:
                params[param_name] = self._default_params[param_name]

        return params

    # ==================== 第三层：文本过滤（占位） ====================

    def filter_text(self, text: str, personality_vec: np.ndarray,
                    cat_name: str) -> Tuple[str, bool]:
        """
        检查LLM生成文本是否符合性格。
        当前为占位实现，完整版需集成情感分类器。
        返回: (文本, 是否通过)
        """
        # 当前阶段：简单关键词检查
        forbidden_words = self._get_forbidden_words(personality_vec)
        for word in forbidden_words:
            if word in text:
                return f"（{cat_name}犹豫了一下，没有开口）", False
        return text, True

    def _get_forbidden_words(self, personality_vec: np.ndarray) -> List[str]:
        """根据性格获取禁用词列表"""
        words = set()
        trait_threshold = 0.7

        forbidden_map = {
            "怯懦": ["最喜欢", "主动过来", "我不怕", "陪我玩", "快点", "过来啊"],
            "傲娇": ["最喜欢你了", "我好想你", "陪我玩嘛", "好开心", "太棒了"],
            "独立": ["陪陪我", "不要走", "好孤单", "我想你"],
            "攻击性": ["爱你", "蹭蹭", "抱抱"],
        }

        for j, trait in enumerate(PERSONALITY_KEYS):
            if personality_vec[j] > trait_threshold:
                words.update(forbidden_map.get(trait, []))

        return list(words)

    # ==================== 调试接口 ====================

    def explain_intent_bias(self, personality_vec: np.ndarray,
                            intent_name: str) -> str:
        """解释某意图对某性格的偏置来源"""
        intent_idx = INTENT_LIST.index(intent_name) if intent_name in INTENT_LIST else -1
        if intent_idx < 0:
            return "未知意图"

        parts = []
        total_bias = 0.0
        for j, trait in enumerate(PERSONALITY_KEYS):
            if personality_vec[j] > 0.01:
                weight = self._intent_trait_matrix.get(trait, {}).get(intent_name, 0.0)
                contribution = weight * personality_vec[j]
                total_bias += contribution
                if abs(contribution) > 0.1:
                    parts.append(f"  {trait}({personality_vec[j]:.2f}) × {weight:.1f} = {contribution:+.2f}")

        return f"{intent_name} 总偏置: {total_bias:+.2f}\n" + "\n".join(parts)

    def get_behavior_param_explanation(self, personality_vec: np.ndarray,
                                       param_name: str) -> str:
        """解释某行为参数的来源"""
        params = self.get_behavior_params(personality_vec)
        value = params.get(param_name, 0)

        parts = [f"{param_name} = {value:.2f}"]
        total = 0.0
        for j, trait in enumerate(PERSONALITY_KEYS):
            if personality_vec[j] > 0.01:
                trait_val = self._behavior_param_table.get(trait, {}).get(param_name, 0)
                if abs(trait_val) > 0.01:
                    contrib = trait_val * personality_vec[j]
                    total += contrib
                    parts.append(f"  {trait}({personality_vec[j]:.2f}) → {trait_val:.2f}")

        return "\n".join(parts)
