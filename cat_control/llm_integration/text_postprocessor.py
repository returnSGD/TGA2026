"""
《猫语心声》 —— 文本后处理与性格过滤（第三层接入）

技术策划案v2 §4.4.3 第三层过滤的完整实现：
- LLM原始输出清洗（去除think标签、截断、去引号）
- 性格关键词检查（接入PersonalityFilter第三层）
- 情感极性一致性检查（预留BERT-tiny分类器接口）
- 回退到模板库（检查失败时）
- DeepSeek-R1 特殊处理：去除 <｜end▁of▁thinking｜>... 标签

设计原则（§4.4.3）:
  - 关键词禁止列表（按性格配置）
  - 情感极性检查：使用轻量情感分类器判断生成文本的情感倾向
    是否与当前情绪状态一致
  - 回退策略：检查失败时使用性格模板文本库
"""

from __future__ import annotations
import re
import sys
import os
from typing import Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import numpy as np

from rl_environment.config import (
    PERSONALITY_KEYS, PERSONALITY_DIM, PERSONALITY_FORBIDDEN_WORDS,
    INTENT_LIST, CAT_CONFIGS,
)

from .config import LLMConfig


# ═══════════════════════════════════════════
#  DeepSeek-R1 输出清洗正则
# ═══════════════════════════════════════════

# R1模型可能在输出中包含  ...  标签
R1_THINK_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL)
R1_THINK_OPEN_PATTERN = re.compile(r'<think>.*', re.DOTALL)

# 多余的空白和引号
MULTI_NEWLINE = re.compile(r'\n{3,}')
QUOTES_PATTERN = re.compile(r'^["\'""'']+|["\'""'']+$')
CITATION_PATTERN = re.compile(r'\[.*?\]')

# 非猫咪独白内容的检测（旁白、解释、思考过程）
NARRATION_PATTERNS = [
    re.compile(p) for p in [
        r'^(输出|生成|独白|内心独白)[:：]',
        r'^(作为|身为).*(猫咪|猫).*[,，]',
        r'^[（(].*[）)]\s*',
        r'^\*.*\*$',           # *动作描述*
    ]
]


def clean_r1_output(raw_text: str) -> str:
    """
    清洗DeepSeek-R1模型的原始输出。

    处理顺序:
    1. 移除 ... 标签（R1的思考过程）
    2. 移除残留的 ... 开头（未闭合标签）
    3. 移除旁白/解释语句
    4. 截取第一句有效文本
    5. 去除多余空白和引号
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # 1. 移除完整的 ... 标签
    text = R1_THINK_PATTERN.sub('', text)

    # 2. 处理未闭合的  标签
    if '<think>' in text:
        text = R1_THINK_OPEN_PATTERN.sub('', text)

    # 3. 移除引用标记
    text = CITATION_PATTERN.sub('', text)

    # 4. 移除旁白/解释前缀
    for pattern in NARRATION_PATTERNS:
        text = pattern.sub('', text)

    # 5. 取第一句或第一段有效内容
    text = _extract_first_utterance(text)

    # 6. 清理空白和引号
    text = MULTI_NEWLINE.sub('\n', text)
    text = QUOTES_PATTERN.sub('', text)
    text = text.strip()

    return text


def _extract_first_utterance(text: str) -> str:
    """提取第一句有效独白（按句号、换行、或长度截断）"""
    if not text:
        return ""

    # 按换行取第一段
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 跳过明显的非独白行
        if line.startswith(('注', '说明', '提示', '注意')):
            continue
        if len(line) >= 2 and not line.startswith('<'):
            return line

    return text


def truncate_to_max_chars(text: str, max_chars: int = 30) -> str:
    """截断到最大字符数（中文约20字=30字符安全上限）"""
    if len(text) <= max_chars:
        return text
    # 尝试在句号处截断
    for sep in ['。', '！', '？', '…', '，', ',', ' ']:
        idx = text.rfind(sep, 0, max_chars)
        if idx > max_chars // 2:
            return text[:idx + 1]
    return text[:max_chars]


class TextPostprocessor:
    """
    LLM输出后处理器 —— 第三层性格过滤。

    处理流程:
    1. 原始输出清洗（R1标签、旁白、引号）
    2. 长度控制（截断到30字符）
    3. 性格关键词检查（接入PERSONALITY_FORBIDDEN_WORDS）
    4. 情感极性一致性检查（预留BERT-tiny接口）
    5. 检查失败 → 回退到模板库

    接入点（技术策划案v2 §4.4.3）:
      在LLM生成心声文本后，通过后处理过滤层确保文本符合性格
    """

    def __init__(self, config: LLMConfig = None,
                 template_library=None):
        self.cfg = config or LLMConfig()
        self._templates = template_library

        # 性格阈值
        self._trait_threshold = 0.7

        # 统计
        self._total_processed = 0
        self._total_cleaned = 0
        self._total_filtered = 0
        self._total_truncated = 0
        self._total_fallback = 0

    def set_template_library(self, template_library):
        self._templates = template_library

    def process(self, raw_text: str,
                personality_vec: np.ndarray,
                cat_id: str = "",
                cat_name: str = "",
                intent: str = "idle_wander",
                fear_value: float = 30.0,
                trust_value: float = 50.0,
                ) -> Tuple[str, bool]:
        """
        后处理LLM原始输出。

        参数:
            raw_text: LLM原始输出文本
            personality_vec: [8] 性格向量
            cat_id: 猫咪ID（回退时查模板库）
            cat_name: 猫咪名字
            intent: 当前意图
            fear_value: 恐惧值
            trust_value: 信任度

        返回: (处理后文本, 是否通过所有检查)
        """
        self._total_processed += 1

        # 1. 清洗原始输出
        cleaned = clean_r1_output(raw_text)
        if cleaned != raw_text:
            self._total_cleaned += 1

        # 2. 长度控制
        if len(cleaned) > 30:
            cleaned = truncate_to_max_chars(cleaned, 30)
            self._total_truncated += 1

        # 空文本直接回退
        if not cleaned or len(cleaned) < 2:
            self._total_fallback += 1
            fallback = self._get_fallback(cat_id, intent, cat_name,
                                          fear_value, trust_value)
            return fallback, False

        # 3. 性格关键词检查
        passed, reason = self._check_personality_keywords(
            cleaned, personality_vec, cat_name
        )
        if not passed:
            self._total_filtered += 1
            self._total_fallback += 1
            fallback = self._get_fallback(cat_id, intent, cat_name,
                                          fear_value, trust_value)
            return fallback, False

        # 4. TODO: 情感极性一致性检查
        # sentiment_ok = self._check_sentiment_consistency(
        #     cleaned, emotion_vector
        # )
        # if not sentiment_ok:
        #     self._total_filtered += 1
        #     self._total_fallback += 1
        #     return self._get_fallback(...), False

        return cleaned, True

    def _check_personality_keywords(self, text: str,
                                     personality_vec: np.ndarray,
                                     cat_name: str = ""
                                     ) -> Tuple[bool, str]:
        """
        检查文本是否包含性格禁用词。

        技术策划案v2 §4.4.3:
          性格向量某维度 > 0.7 时，该维度的所有禁用词生效

        返回: (通过?, 失败原因)
        """
        for j, trait in enumerate(PERSONALITY_KEYS):
            if personality_vec[j] > self._trait_threshold:
                forbidden = PERSONALITY_FORBIDDEN_WORDS.get(trait, [])
                for word in forbidden:
                    if word in text:
                        return False, (
                            f"包含'{trait}'性格禁用词: '{word}'"
                        )
        return True, "ok"

    def _check_sentiment_consistency(self, text: str,
                                      emotion_vector: np.ndarray
                                      ) -> bool:
        """
        [预留接口] 情感极性一致性检查。

        使用BERT-tiny微调模型判断生成文本的情感倾向，
        与当前情绪状态（恐惧值、信任度等）进行一致性校验。

        例如：恐惧值70但生成文本极性为"欢快" → 不一致

        当前以规则替代：检测关键词级别的冲突。
        """
        if emotion_vector is None or len(emotion_vector) < 5:
            return True

        fear = emotion_vector[1]  # 0-1

        # 高恐惧时不允许出现轻松/欢快的关键词
        if fear > 0.6:
            high_fear_forbidden = [
                "好开心", "太棒了", "好舒服", "真快乐",
                "好好玩", "兴奋", "哈哈哈", "一点都不怕",
            ]
            for word in high_fear_forbidden:
                if word in text:
                    return False

        # 低恐惧时不需要检查（可以自由表达）
        return True

    def _get_fallback(self, cat_id: str, intent: str, cat_name: str,
                      fear: float = 30.0, trust: float = 50.0) -> str:
        """获取回退文本"""
        if self._templates is not None:
            return self._templates.get_template_for_emotion(
                cat_id, intent, fear, trust
            )
        if cat_name:
            return f"（{cat_name}犹豫了一下，没有开口）"
        return "……"

    # ═══════════════════════════════════════════
    #  统计
    # ═══════════════════════════════════════════

    @property
    def stats(self) -> Dict:
        return {
            "total_processed": self._total_processed,
            "total_cleaned": self._total_cleaned,
            "total_filtered": self._total_filtered,
            "total_truncated": self._total_truncated,
            "total_fallback": self._total_fallback,
            "pass_rate": (
                1.0 - self._total_fallback / max(1, self._total_processed)
            ),
            "clean_rate": (
                self._total_cleaned / max(1, self._total_processed)
            ),
        }

    def get_report(self) -> str:
        s = self.stats
        return (
            f"══════════ 文本后处理报告 ══════════\n"
            f"  总处理:    {s['total_processed']}\n"
            f"  清洗:      {s['total_cleaned']} "
            f"({s['clean_rate']:.1%})\n"
            f"  性格过滤:  {s['total_filtered']}\n"
            f"  截断:      {s['total_truncated']}\n"
            f"  回退:      {s['total_fallback']}\n"
            f"  通过率:    {s['pass_rate']:.1%}\n"
            f"══════════════════════════════════════"
        )

    def reset_stats(self):
        self._total_processed = 0
        self._total_cleaned = 0
        self._total_filtered = 0
        self._total_truncated = 0
        self._total_fallback = 0


# ═══════════════════════════════════════════
#  R1输出清洗测试
# ═══════════════════════════════════════════

if __name__ == "__main__":
    # 测试R1输出清洗
    test_cases = [
        # R1典型输出（含think标签）
        ("<think>嗯，用户让我用猫的视角表达感受。现在我是小雪了，"
         "我是一只胆小的波斯猫，很害怕人类。外面好像有人在靠近，"
         "我应该表达出不安。</think>\n这个人……他走得很慢，也许不是坏人……",
         "这个人……他走得很慢，也许不是坏人……"),

        # 带引号
        ('"今天也是普通的一天。"', "今天也是普通的一天。"),

        # 旁白前缀
        ("内心独白：今天天气不错。", "今天天气不错。"),

        # 多行输出
        ("第一句。\n第二句。\n第三句。", "第一句。"),

        # 空输出
        ("", ""),

        # 未闭合的think标签
        ("<think>用户让我...\n这个人很温柔。", "这个人很温柔。"),
    ]

    print("=== R1输出清洗测试 ===\n")
    for raw, expected in test_cases:
        result = clean_r1_output(raw)
        status = "PASS" if result == expected else f"FAIL (got: '{result}')"
        print(f"{status}  raw: '{raw[:50]}...'")
        print(f"   expected: '{expected}'")
        print(f"   result:   '{result}'\n")

    # 测试性格关键词过滤
    print("=== 性格关键词过滤测试 ===\n")
    pp = TextPostprocessor()

    # 小雪（怯懦0.9）
    xiaoxue_personality = np.array(
        [0.0, 0.9, 0.1, 0.2, 0.1, 0.0, 0.1, 0.3], dtype=np.float32
    )

    # 应该通过的文本
    text_ok = "这个人……好像没有要抓我……"
    result, passed = pp.process(text_ok, xiaoxue_personality,
                                cat_id="xiaoxue", cat_name="小雪")
    print(f"{'[PASS]' if passed else '[FAIL]'} 通过: '{text_ok}' -> '{result}'")

    # 应该被拦截的文本（含"最喜欢"——怯懦猫禁用词）
    text_bad = "我最喜欢这个人了！"
    result, passed = pp.process(text_bad, xiaoxue_personality,
                                cat_id="xiaoxue", cat_name="小雪")
    print(f"{'[PASS]' if not passed else '[FAIL]'} 拦截: '{text_bad}' -> '{result}'")

    # 奥利奥（傲娇0.8）
    oreo_personality = np.array(
        [0.8, 0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.7], dtype=np.float32
    )
    text_oreo_bad = "最喜欢你了！陪我玩嘛！"
    result, passed = pp.process(text_oreo_bad, oreo_personality,
                                cat_id="oreo", cat_name="奥利奥")
    print(f"{'[PASS]' if not passed else '[FAIL]'} 拦截(傲娇): '{text_oreo_bad}' -> '{result}'")

    print(f"\n{pp.get_report()}")
