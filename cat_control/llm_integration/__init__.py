"""
《猫语心声》 —— LLM 心声生成与集成模块

技术策划案v2 §4.5 的完整实现：
- 本地LLM推理服务 (llama.cpp + DeepSeek-R1-Distill-Qwen-1.5B-Q4_0)
- 缓存与降级机制
- 心声生成提示词构建
- 文本后处理与性格过滤（第三层接入）
- 端到端联调: RL决策 → 行为树 → LLM心声
"""

from .config import LLMConfig
from .llm_service import LLMService
from .prompt_builder import PromptBuilder
from .template_library import TemplateLibrary
from .cache_fallback import CacheFallbackManager
from .text_postprocessor import TextPostprocessor
from .monologue_generator import MonologueGenerator

__all__ = [
    "LLMConfig",
    "LLMService",
    "PromptBuilder",
    "TemplateLibrary",
    "CacheFallbackManager",
    "TextPostprocessor",
    "MonologueGenerator",
]
