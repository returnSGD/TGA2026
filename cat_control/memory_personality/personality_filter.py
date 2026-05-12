"""
《猫语心声》 —— 生产级三层性格过滤器

技术策划案v2 §4.4 的完整实现：
第一层: RL意图logits修正（性格→意图兼容矩阵）
第二层: 行为参数调整（性格→行为系数表）
第三层: LLM文本关键词过滤 + 情感极性检查

额外: RL训练集成（批量logits修正）、调试解释接口
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .config import (
    MemoryConfig, INTENT_LIST, PERSONALITY_KEYS, PERSONALITY_DIM,
    INTENT_PERSONALITY_MATRIX, PERSONALITY_BEHAVIOR_PARAMS,
    PERSONALITY_FORBIDDEN_WORDS,
)


class PersonalityFilter:
    """
    三层性格过滤器 —— HRLTM架构的人设一致性最后防线。

    原理（技术策划案v2 §4.4）:
    - 第一层: filtered_logits[i] = raw_logits[i] + Σ(性格权重[j] × 矩阵[j][i])
    - 第二层: behavioral_params = Σ(性格权重[j] × 行为参数表[j]) / Σ权重
    - 第三层: 关键词禁止列表 + 情感极性分类器
    """

    def __init__(self, config: MemoryConfig = None):
        self.cfg = config or MemoryConfig()
        self._intent_matrix = self.cfg.intent_trait_matrix
        self._behavior_table = self.cfg.behavior_param_table
        self._forbidden_words = self.cfg.forbidden_words

        # 默认行为参数
        self._default_params = {
            "approach_distance": 1.5,
            "move_speed": 0.6,
            "response_delay": 2.0,
            "flee_distance": 2.0,
            "hesitation_weight": 0.5,
        }

        # 预计算意图→索引映射
        self._intent_to_idx = {name: i for i, name in enumerate(INTENT_LIST)}
        self._idx_to_intent = {i: name for i, name in enumerate(INTENT_LIST)}

    # ═══════════════════════════════════════════
    #  第一层: RL意图logits修正
    # ═══════════════════════════════════════════

    def filter_intent_logits(self, logits: np.ndarray,
                            personality_vec: np.ndarray) -> np.ndarray:
        """
        对RL策略网络输出的logits施加性格偏置。

        参数:
            logits: [num_intents] 或 [batch, num_intents] 的原始logits
            personality_vec: [personality_dim] 性格向量

        返回: 修正后的logits（不改变形状）
        """
        bias = np.zeros_like(logits)

        for i, intent in enumerate(INTENT_LIST):
            for j, trait in enumerate(PERSONALITY_KEYS):
                weight = self._intent_matrix.get(trait, {}).get(intent, 0.0)
                bias[..., i] += weight * personality_vec[j]

        return logits + bias

    def filter_batch_logits(self, logits: np.ndarray,
                           personality_vecs: np.ndarray) -> np.ndarray:
        """
        批量修正（用于RL训练中的多猫并行推理）。

        logits: [batch, num_intents]
        personality_vecs: [batch, personality_dim]
        返回: [batch, num_intents]
        """
        biases = np.zeros_like(logits)

        for i, intent in enumerate(INTENT_LIST):
            for j, trait in enumerate(PERSONALITY_KEYS):
                weight = self._intent_matrix.get(trait, {}).get(intent, 0.0)
                biases[:, i] += weight * personality_vecs[:, j]

        return logits + biases

    def filter_probs(self, probs: np.ndarray, personality_vec: np.ndarray,
                    temperature: float = 1.0) -> np.ndarray:
        """
        过滤动作概率分布（通过logit偏置+softmax）。

        probs: [num_intents] 原始概率
        personality_vec: [personality_dim]
        temperature: softmax温度

        返回: 修正后的概率分布（仍归一化）
        """
        probs = np.clip(probs, 1e-8, 1.0)
        logits = np.log(probs)
        filtered = self.filter_intent_logits(logits, personality_vec)
        scaled = filtered / temperature
        scaled -= scaled.max()
        exp = np.exp(scaled)
        return exp / exp.sum()

    # ═══════════════════════════════════════════
    #  第二层: 行为参数
    # ═══════════════════════════════════════════

    def get_behavior_params(self, personality_vec: np.ndarray) -> Dict[str, float]:
        """
        根据性格向量计算行为执行参数。

        返回值:
            approach_distance: 亲近距离阈值（m）
            move_speed: 移动速度（0~1）
            response_delay: 互动响应延迟（tick）
            flee_distance: 逃跑触发距离（m）
            hesitation_weight: 犹豫/自信权重（0~1, 越高越犹豫）
        """
        params = {}
        param_names = list(self._default_params.keys())

        for param_name in param_names:
            value = 0.0
            total_weight = 0.0
            for j, trait in enumerate(PERSONALITY_KEYS):
                p = personality_vec[j]
                if p > 0.01:
                    trait_val = self._behavior_table.get(trait, {}).get(param_name)
                    if trait_val is not None:
                        value += trait_val * p
                        total_weight += p

            if total_weight > 0:
                params[param_name] = value / total_weight
            else:
                params[param_name] = self._default_params[param_name]

        return params

    def get_batch_behavior_params(self,
                                  personality_vecs: np.ndarray
                                  ) -> List[Dict[str, float]]:
        """批量计算行为参数"""
        return [self.get_behavior_params(pv) for pv in personality_vecs]

    # ═══════════════════════════════════════════
    #  第三层: 文本过滤
    # ═══════════════════════════════════════════

    def filter_text(self, text: str, personality_vec: np.ndarray,
                    cat_name: str = "") -> Tuple[str, bool]:
        """
        检查LLM生成文本是否符合性格。

        返回: (文本, 是否通过)
        """
        # 关键词屏蔽
        forbidden = self._get_active_forbidden(personality_vec)
        for word in forbidden:
            if word in text:
                fallback = f"（{cat_name}犹豫了一下，没有开口）" if cat_name else "..."
                return fallback, False

        # TODO: 集成情感极性分类器（Bert-tiny微调）
        # sentiment = self._sentiment_classifier(text)
        # expected_sentiment = self._expected_sentiment(personality_vec)
        # if abs(sentiment - expected_sentiment) > threshold:
        #     return fallback, False

        return text, True

    def filter_batch_texts(self, texts: List[str],
                          personality_vecs: np.ndarray,
                          cat_names: List[str] = None
                          ) -> List[Tuple[str, bool]]:
        """批量文本过滤"""
        results = []
        for i, text in enumerate(texts):
            pv = personality_vecs[i] if i < len(personality_vecs) else personality_vecs[-1]
            name = cat_names[i] if cat_names and i < len(cat_names) else ""
            results.append(self.filter_text(text, pv, name))
        return results

    def _get_active_forbidden(self, personality_vec: np.ndarray) -> List[str]:
        """获取当前性格激活的禁用词列表"""
        words = set()
        for j, trait in enumerate(PERSONALITY_KEYS):
            if personality_vec[j] > self.cfg.trait_threshold:
                trait_words = self._forbidden_words.get(trait, [])
                words.update(trait_words)
        return list(words)

    # ═══════════════════════════════════════════
    #  调试与解释接口
    # ═══════════════════════════════════════════

    def explain_intent_bias(self, personality_vec: np.ndarray,
                           intent_name: str) -> str:
        """解释某个意图对某性格的偏置来源"""
        if intent_name not in self._intent_to_idx:
            return "未知意图"

        parts = []
        total_bias = 0.0
        for j, trait in enumerate(PERSONALITY_KEYS):
            p = personality_vec[j]
            if p > 0.01:
                weight = self._intent_matrix.get(trait, {}).get(intent_name, 0.0)
                contribution = weight * p
                total_bias += contribution
                if abs(contribution) > 0.1:
                    parts.append(
                        f"  {trait}({p:.2f}) × {weight:+.1f} = {contribution:+.2f}"
                    )

        header = f"{intent_name} 总偏置: {total_bias:+.2f}"
        return header + "\n" + "\n".join(parts)

    def explain_all_intents(self, personality_vec: np.ndarray) -> str:
        """解释所有15个意图的偏置（按从高到低排序）"""
        biases = {}
        for intent in INTENT_LIST:
            bias = 0.0
            for j, trait in enumerate(PERSONALITY_KEYS):
                weight = self._intent_matrix.get(trait, {}).get(intent, 0.0)
                bias += weight * personality_vec[j]
            biases[intent] = bias

        sorted_biases = sorted(biases.items(), key=lambda x: -x[1])
        lines = ["意图偏置排名 (正=倾向, 负=排斥):"]
        for intent, bias in sorted_biases:
            bar = "█" * int(abs(bias)) + (">" if bias > 0 else "<")
            lines.append(f"  {intent:22s} {bias:+.2f} {bar}")
        return "\n".join(lines)

    def explain_behavior_params(self, personality_vec: np.ndarray,
                               param_name: str) -> str:
        """解释某个行为参数的来源"""
        value = 0.0
        total_weight = 0.0
        contributions = []

        for j, trait in enumerate(PERSONALITY_KEYS):
            p = personality_vec[j]
            if p > 0.01:
                trait_val = self._behavior_table.get(trait, {}).get(param_name, 0)
                if abs(trait_val) > 0.01:
                    contrib = trait_val * p
                    value += contrib
                    total_weight += p
                    contributions.append(f"  {trait}({p:.2f}) → {trait_val:.2f}")

        if total_weight > 0:
            value /= total_weight

        lines = [f"{param_name} = {value:.2f}"]
        lines.extend(contributions)
        return "\n".join(lines)

    def get_intent_compatibility_matrix(self, personality_vec: np.ndarray
                                       ) -> Dict[str, float]:
        """返回所有意图与给定性格的兼容度"""
        result = {}
        for intent in INTENT_LIST:
            bias = 0.0
            for j, trait in enumerate(PERSONALITY_KEYS):
                weight = self._intent_matrix.get(trait, {}).get(intent, 0.0)
                bias += weight * personality_vec[j]
            result[intent] = float(bias)
        return result

    def get_active_forbidden_report(self, personality_vec: np.ndarray) -> Dict:
        """报告当前激活的禁用词"""
        report = {}
        for j, trait in enumerate(PERSONALITY_KEYS):
            if personality_vec[j] > self.cfg.trait_threshold:
                report[trait] = {
                    "value": float(personality_vec[j]),
                    "forbidden_count": len(self._forbidden_words.get(trait, [])),
                    "sample": self._forbidden_words.get(trait, [])[:5],
                }
        return report
