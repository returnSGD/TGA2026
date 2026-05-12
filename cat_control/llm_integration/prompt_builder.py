"""
《猫语心声》 —— 心声生成提示词构建器

技术策划案v2 §4.5.2 LLM输入格式的实现：
- 系统指令: 猫咪身份、性格、语气要求
- 用户消息: 当前状态、情绪、意图、记忆、场景

核心设计原则:
- LLM仅负责文本渲染，不参与行为决策
- 提示词严格绑定当前意图和情绪
- 记忆上下文通过向量检索注入
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .config import (
    LLMConfig,
    SYSTEM_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE,
    INTENT_DESCRIPTIONS,
    PLAYER_ACTION_DESCRIPTIONS,
    TIME_DESCRIPTIONS,
    PERSONALITY_TONE,
    PERSONALITY_CORE_RULE,
)
from rl_environment.config import (
    INTENT_LIST,
    PERSONALITY_KEYS,
    PERSONALITY_DIM,
    CAT_CONFIGS,
)


def _get_dominant_trait(personality_vec: np.ndarray) -> Tuple[str, float]:
    """获取性格向量中权重最大的维度"""
    idx = np.argmax(personality_vec)
    return PERSONALITY_KEYS[idx], float(personality_vec[idx])


def _personality_to_description(personality_vec: np.ndarray) -> str:
    """将性格向量转为自然语言描述"""
    traits = []
    for j, trait in enumerate(PERSONALITY_KEYS):
        val = personality_vec[j]
        if val > 0.6:
            traits.append(f"非常{trait}")
        elif val > 0.3:
            traits.append(f"比较{trait}")
        elif val > 0.1:
            traits.append(f"有一点{trait}")
    if not traits:
        return "性格平和"
    return "、".join(traits)


def _get_tone_requirement(personality_vec: np.ndarray) -> str:
    """获取主要性格维度的语气要求"""
    dominant_trait, weight = _get_dominant_trait(personality_vec)
    if weight < 0.3:
        return "自然平和，像一只普通猫咪"
    return PERSONALITY_TONE.get(dominant_trait, "自然表达")


def _get_core_rule(personality_vec: np.ndarray) -> str:
    """获取性格核心行为准则"""
    dominant_trait, weight = _get_dominant_trait(personality_vec)
    if weight < 0.3:
        return "是一只普通的猫咪，自然地表达自己的想法"
    return PERSONALITY_CORE_RULE.get(dominant_trait, "自然地做一只猫咪")


def _format_memories(memories: List) -> str:
    """格式化记忆列表为提示词字符串"""
    if not memories:
        return "（暂无相关记忆）"

    lines = []
    for i, mem in enumerate(memories[:3]):
        # MemoryItem 或 dict
        if hasattr(mem, 'desc'):
            desc = mem.desc
        elif isinstance(mem, dict):
            desc = mem.get("desc", str(mem))
        else:
            desc = str(mem)
        lines.append(f"{i+1}. {desc}")
    return "\n".join(lines)


def _get_time_phase(hour: float = 12.0) -> str:
    """游戏时间→时间阶段描述"""
    if hour < 6:
        return "night"
    elif hour < 10:
        return "morning"
    elif hour < 17:
        return "afternoon"
    elif hour < 21:
        return "evening"
    return "night"


@dataclass
class PromptContext:
    """构建提示词所需的完整上下文"""
    cat_id: str
    cat_name: str
    cat_breed: str = ""
    cat_backstory: str = ""

    # 性格
    personality_vec: np.ndarray = None

    # 当前情绪
    hunger: float = 30.0
    fear: float = 30.0
    curiosity: float = 50.0
    comfort: float = 50.0
    social: float = 30.0
    trust: float = 50.0

    # 当前意图
    intent: str = "idle_wander"

    # 玩家行为
    player_action: str = "none"

    # 记忆
    memories: List = None

    # 场景
    scene_desc: str = "猫咖大厅，一切如常"
    time_of_day: float = 12.0

    def __post_init__(self):
        if self.personality_vec is None:
            self.personality_vec = np.zeros(PERSONALITY_DIM, dtype=np.float32)
        if self.memories is None:
            self.memories = []


class PromptBuilder:
    """
    心声生成提示词构建器。

    根据猫咪状态、意图、情绪、记忆构建结构化的LLM输入。
    输出格式遵循技术策划案v2 §4.5.2 的模板定义。
    """

    def __init__(self, config: LLMConfig = None):
        self.cfg = config or LLMConfig()

    def build_system_prompt(self, ctx: PromptContext) -> str:
        """构建系统指令（猫咪身份 + 性格约束）"""
        if not ctx.cat_breed:
            cat_cfg = CAT_CONFIGS.get(ctx.cat_id, {})
            ctx.cat_breed = cat_cfg.get("breed", "猫咪")
            ctx.cat_backstory = cat_cfg.get("backstory", "")

        personality_desc = _personality_to_description(ctx.personality_vec)
        tone = _get_tone_requirement(ctx.personality_vec)
        core_rule = _get_core_rule(ctx.personality_vec)

        return self.cfg.system_prompt_template.format(
            name=ctx.cat_name,
            breed=ctx.cat_breed,
            personality_desc=personality_desc,
            personality_core_rule=core_rule,
            backstory=ctx.cat_backstory,
            tone_requirement=tone,
        )

    def build_user_prompt(self, ctx: PromptContext) -> str:
        """构建用户消息（当前状态 + 意图 + 记忆 + 场景）"""
        intent_desc = INTENT_DESCRIPTIONS.get(ctx.intent, "发呆")
        player_desc = PLAYER_ACTION_DESCRIPTIONS.get(
            ctx.player_action, "待在房间里"
        )
        time_phase = _get_time_phase(ctx.time_of_day)
        time_desc = TIME_DESCRIPTIONS.get(time_phase, "午后时光")

        memory_text = _format_memories(ctx.memories)

        return self.cfg.user_prompt_template.format(
            hunger=f"{ctx.hunger:.0f}",
            fear=f"{ctx.fear:.0f}",
            curiosity=f"{ctx.curiosity:.0f}",
            comfort=f"{ctx.comfort:.0f}",
            social=f"{ctx.social:.0f}",
            trust=f"{ctx.trust:.0f}",
            intent_desc=intent_desc,
            player_action_desc=player_desc,
            memory_context=memory_text,
            scene_desc=ctx.scene_desc,
            time_desc=time_desc,
        )

    def build_full_prompt(self, ctx: PromptContext) -> str:
        """
        构建完整聊天格式提示词（用于DeepSeek-R1-Distill-Qwen的chat template）。

        采用对话格式:
        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {user_prompt}<|im_end|>
        <|im_start|>assistant
        """
        system = self.build_system_prompt(ctx)
        user = self.build_user_prompt(ctx)

        # DeepSeek-R1-Distill-Qwen 使用 Qwen chat template
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def build_text_completion_prompt(self, ctx: PromptContext) -> str:
        """
        构建纯文本补全格式（用于部分不支持chat template的推理方式）。
        """
        system = self.build_system_prompt(ctx)
        user = self.build_user_prompt(ctx)

        return (
            f"指令：{system}\n\n"
            f"当前情景：{user}\n\n"
            f"内心独白："
        )

    def build_from_cat_state(self, cat_state, intent: str = None,
                            player_action: str = "none",
                            memories: List = None,
                            scene_desc: str = "猫咖大厅") -> PromptContext:
        """
        从CatState对象快速构建PromptContext（用于RL Agent集成）。
        """
        cat_cfg = CAT_CONFIGS.get(cat_state.cat_id, {})

        ctx = PromptContext(
            cat_id=getattr(cat_state, 'cat_id', 'unknown'),
            cat_name=getattr(cat_state, 'name', '猫咪'),
            cat_breed=cat_cfg.get("breed", ""),
            cat_backstory=cat_cfg.get("backstory", ""),
            personality_vec=getattr(cat_state, 'personality_vector',
                                   np.zeros(PERSONALITY_DIM, dtype=np.float32)),
            hunger=float(getattr(cat_state, 'emotion_vector',
                        np.zeros(5))[0]) * 100 if hasattr(cat_state, 'emotion_vector') else 30.0,
            fear=float(getattr(cat_state, 'emotion_vector',
                       np.zeros(5))[1]) * 100 if hasattr(cat_state, 'emotion_vector') else 30.0,
            curiosity=float(getattr(cat_state, 'emotion_vector',
                           np.zeros(5))[2]) * 100 if hasattr(cat_state, 'emotion_vector') else 50.0,
            comfort=float(getattr(cat_state, 'emotion_vector',
                          np.zeros(5))[3]) * 100 if hasattr(cat_state, 'emotion_vector') else 50.0,
            social=float(getattr(cat_state, 'emotion_vector',
                         np.zeros(5))[4]) * 100 if hasattr(cat_state, 'emotion_vector') else 30.0,
            trust=getattr(cat_state, 'trust_level', 50.0),
            intent=intent or "idle_wander",
            player_action=player_action,
            memories=memories or [],
            scene_desc=scene_desc,
        )
        return ctx

    @staticmethod
    def estimate_tokens(prompt: str) -> int:
        """粗略估算prompt的token数（中文约1.5字/token）"""
        # 中英文混合粗略估算
        chinese_chars = sum(1 for c in prompt if '一' <= c <= '鿿')
        other_chars = len(prompt) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)
