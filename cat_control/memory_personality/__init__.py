"""
《猫语心声》 —— 记忆系统与性格过滤器（阶段三：第9-10周）

HRLTM架构核心模块：
- 两级记忆系统（工作记忆 + 长期记忆 + 向量语义检索）
- 三层性格过滤器（意图修正 + 行为参数 + 文本过滤）
- 记忆→RL状态注入桥梁
- 性格差异验证工具

技术栈：
- 向量数据库: sqlite-vec / Chroma / numpy fallback
- 语义嵌入: Sentence Transformer (all-MiniLM-L6-v2, 128维)
- 记忆压缩: 时间衰减 + 摘要向量
"""

from .config import MemoryConfig
from .vector_store import VectorStore, NumpyVectorStore
from .embedding import EmbeddingService
from .memory_manager import MemoryManager
from .personality_filter import PersonalityFilter
from .memory_rl_bridge import MemoryRLBridge
