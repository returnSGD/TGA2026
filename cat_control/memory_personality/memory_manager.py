"""
《猫语心声》 —— 生产级两级记忆系统

技术策划案v2 §4.3 的完整实现：
- 工作记忆: 固定窗口环形队列（20条），为RL提供短期上下文
- 长期记忆: 优先级队列（500条）+ 向量语义检索 + 时间衰减
- 记忆压缩: 旧记忆 → 摘要向量，释放容量
- 真实语义嵌入: Sentence Transformer + 向量数据库
"""

from __future__ import annotations
import os
import time
import heapq
import random
import numpy as np
from collections import deque
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from .config import MemoryConfig, EVENT_IMPORTANCE_MID
from .vector_store import VectorStore, NumpyVectorStore, create_vector_store
from .embedding import EmbeddingService, get_embedding_service


@dataclass
class MemoryItem:
    """记忆单元"""
    memory_id: str
    desc: str
    timestamp: float            # 游戏tick
    importance: float           # 0.0 ~ 10.0
    embedding: np.ndarray       # 128维语义向量
    event_type: str = "daily"   # feed/pet/scare/social/milestone
    metadata: Dict = field(default_factory=dict)

    def __lt__(self, other: 'MemoryItem') -> bool:
        return self.importance > other.importance  # 大顶堆

    def to_dict(self) -> Dict:
        return {
            "memory_id": self.memory_id,
            "desc": self.desc,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "event_type": self.event_type,
            "metadata": self.metadata,
        }


class MemoryManager:
    """
    生产级两级记忆系统。

    [工作记忆] 环形队列 20条
      ├── 存储最近交互，FIFO淘汰
      ├── 为RL提供短期决策上下文
      └── 高重要性记忆自动进入长期记忆

    [长期记忆] 优先级队列 500条 + 向量数据库
      ├── 按importance排序，低分淘汰
      ├── 语义向量存入 VectorStore
      ├── 时间衰减: score = importance × time_decay
      └── 旧记忆压缩: 多→1条摘要
    """

    def __init__(self, config: MemoryConfig = None,
                 vector_store: VectorStore = None,
                 embed_service: EmbeddingService = None):
        self.cfg = config or MemoryConfig()

        # 向量数据库
        self.vector_store = vector_store or create_vector_store(
            backend=self.cfg.vector_db_backend,
            embed_dim=self.cfg.embed_dim,
        )

        # 嵌入服务
        self.embed_service = embed_service or get_embedding_service(
            target_dim=self.cfg.embed_dim,
        )

        # 工作记忆（环形队列）
        self.working_memory: deque[MemoryItem] = deque(
            maxlen=self.cfg.work_memory_size
        )

        # 长期记忆（大顶堆）
        self.long_term_memory: List[MemoryItem] = []

        # 统计
        self.total_stored = 0
        self.total_retrieved = 0
        self.total_compressed = 0
        self._id_counter = 0

    # ═══════════════════════════════════════════
    #  记忆存储
    # ═══════════════════════════════════════════

    def add_memory(self, desc: str, event_type: str = "daily",
                   timestamp: float = 0.0, importance: float = None,
                   metadata: Dict = None) -> MemoryItem:
        """
        添加一条记忆。

        参数:
            desc: 记忆描述文本（如"玩家轻声安抚我，放下零食后离开"）
            event_type: 事件类型（feed/pet/scare/social/milestone）
            timestamp: 游戏tick
            importance: 重要性（None则根据event_type自动计算）
            metadata: 附加信息
        """
        if importance is None:
            importance = self._calc_importance(event_type, desc)

        # 生成语义嵌入
        embedding = self.embed_service.encode(desc)

        item = MemoryItem(
            memory_id=f"mem_{self._id_counter}",
            desc=desc,
            timestamp=timestamp,
            importance=importance,
            embedding=embedding,
            event_type=event_type,
            metadata=metadata or {},
        )
        self._id_counter += 1

        # 写入工作记忆
        self.working_memory.append(item)

        # 重要性超阈值 → 长期记忆 + 向量数据库
        if importance >= self.cfg.importance_threshold:
            heapq.heappush(self.long_term_memory, item)
            self.vector_store.add(
                item.memory_id, embedding,
                metadata=item.to_dict(),
            )

            # 容量控制
            while len(self.long_term_memory) > self.cfg.long_memory_cap:
                evicted = heapq.heappop(self.long_term_memory)
                self.vector_store.delete(evicted.memory_id)

        self.total_stored += 1
        return item

    def _calc_importance(self, event_type: str, desc: str = "") -> float:
        """根据事件类型计算基础重要性 + 小幅随机抖动"""
        base = EVENT_IMPORTANCE_MID.get(event_type, 4.0)
        # 文本长度暗示信息量
        if len(desc) > 50:
            base = min(10.0, base + 0.5)
        return base + random.uniform(-0.3, 0.3)

    # ═══════════════════════════════════════════
    #  记忆检索
    # ═══════════════════════════════════════════

    def retrieve_by_query(self, query_vector: np.ndarray,
                         top_k: int = 3) -> List[MemoryItem]:
        """
        基于语义相似度检索最相关记忆。

        query_vector: 由当前情绪+环境构建的查询向量（10维，会被投影到128维）
        返回: Top-K最相关的MemoryItem列表
        """
        # 查询向量投影到嵌入空间（如果维度不匹配）
        if query_vector.shape[0] != self.cfg.embed_dim:
            full_query = np.zeros(self.cfg.embed_dim, dtype=np.float32)
            full_query[:len(query_vector)] = query_vector
            query_vector = full_query

        results = self.vector_store.search(query_vector, top_k=top_k)
        self.total_retrieved += len(results)

        items = []
        for mem_id, sim, meta in results:
            # 从长期记忆中查找完整MemoryItem
            for mem in self.long_term_memory:
                if mem.memory_id == mem_id:
                    items.append(mem)
                    break

        return items

    def retrieve_by_event(self, event_type: str, limit: int = 10) -> List[MemoryItem]:
        """按事件类型检索（用于调试/分析）"""
        results = [m for m in self.long_term_memory if m.event_type == event_type]
        results.sort(key=lambda m: -m.importance)
        return results[:limit]

    def retrieve_recent(self, k: int = 5) -> List[MemoryItem]:
        """获取最近k条工作记忆"""
        return list(self.working_memory)[-k:]

    def get_memory_embeddings(self, query_vector: np.ndarray,
                             top_k: int = 3) -> List[np.ndarray]:
        """
        检索Top-K记忆嵌入向量（用于RL状态注入）。
        不足则零向量填充。
        """
        items = self.retrieve_by_query(query_vector, top_k)
        result = []
        for i in range(top_k):
            if i < len(items) and items[i].embedding is not None:
                result.append(items[i].embedding.copy())
            else:
                result.append(np.zeros(self.cfg.embed_dim, dtype=np.float32))
        return result

    def get_recent_memory_embeddings(self, k: int = 3) -> List[np.ndarray]:
        """获取最近k条工作记忆的嵌入（不基于语义检索）"""
        recent = self.retrieve_recent(k)
        result = []
        for i in range(k):
            if i < len(recent):
                result.append(recent[i].embedding.copy())
            else:
                result.append(np.zeros(self.cfg.embed_dim, dtype=np.float32))
        return result

    # ═══════════════════════════════════════════
    #  记忆压缩
    # ═══════════════════════════════════════════

    def compress_old_memories(self, current_time: float) -> int:
        """
        压缩超过阈值天数的旧记忆。

        多→1：对同event_type的旧记忆，取嵌入均值生成一条摘要。
        """
        threshold = current_time - self.cfg.compress_age_days * 144
        old_mems = [m for m in self.long_term_memory if m.timestamp < threshold]

        if len(old_mems) < self.cfg.compress_min_count:
            return 0

        # 按事件类型分组
        groups: Dict[str, List[MemoryItem]] = {}
        for mem in old_mems:
            groups.setdefault(mem.event_type, []).append(mem)

        compressed_count = 0
        for event_type, group in groups.items():
            if len(group) < 3:
                continue

            # 生成摘要嵌入
            embeddings = [m.embedding for m in group if m.embedding is not None]
            if not embeddings:
                continue
            summary_embed = np.mean(embeddings, axis=0)
            summary_embed = summary_embed / (np.linalg.norm(summary_embed) + 1e-8)

            # 创建摘要记忆
            descs = [m.desc for m in group[:3]]
            summary_item = MemoryItem(
                memory_id=f"summary_{self._id_counter}",
                desc=f"[摘要·{event_type}] {'; '.join(descs)}...",
                timestamp=current_time,
                importance=min(m.importance for m in group),
                embedding=summary_embed,
                event_type="summary",
                metadata={"compressed_from": len(group), "event_types": event_type},
            )
            self._id_counter += 1

            # 删除旧记忆，添加摘要
            for mem in group:
                self.long_term_memory.remove(mem)
                self.vector_store.delete(mem.memory_id)
            heapq.heapify(self.long_term_memory)

            heapq.heappush(self.long_term_memory, summary_item)
            self.vector_store.add(summary_item.memory_id, summary_embed,
                                 metadata=summary_item.to_dict())
            compressed_count += len(group)

        self.total_compressed += compressed_count
        return compressed_count

    # ═══════════════════════════════════════════
    #  时间衰减重评分
    # ═══════════════════════════════════════════

    def apply_time_decay(self, current_time: float):
        """
        对所有长期记忆应用时间衰减。

        score = base_importance × max(0.1, 1 - Δt / max_ttl)

        衰减后的重要性会重新排序优先级队列。
        """
        max_ttl = self.cfg.time_decay_max_ttl

        for mem in self.long_term_memory:
            dt = current_time - mem.timestamp
            decay = max(self.cfg.time_decay_min, 1.0 - dt / max_ttl)
            mem.importance = mem.importance * decay

        heapq.heapify(self.long_term_memory)

        # 淘汰衰减到极低分的记忆
        while (self.long_term_memory and
               self.long_term_memory[0].importance < 1.0):
            evicted = heapq.heappop(self.long_term_memory)
            self.vector_store.delete(evicted.memory_id)

    # ═══════════════════════════════════════════
    #  统计与导出
    # ═══════════════════════════════════════════

    @property
    def size_working(self) -> int:
        return len(self.working_memory)

    @property
    def size_long_term(self) -> int:
        return len(self.long_term_memory)

    @property
    def size_vector_db(self) -> int:
        return self.vector_store.count()

    def summary(self) -> str:
        return (
            f"记忆系统: 工作记忆 {self.size_working}/{self.cfg.work_memory_size}, "
            f"长期记忆 {self.size_long_term}/{self.cfg.long_memory_cap}, "
            f"向量库 {self.size_vector_db}, "
            f"总计存储 {self.total_stored}, 检索 {self.total_retrieved}, "
            f"压缩 {self.total_compressed}"
        )

    def get_event_distribution(self) -> Dict[str, int]:
        """长期记忆的事件类型分布"""
        dist = {}
        for mem in self.long_term_memory:
            dist[mem.event_type] = dist.get(mem.event_type, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))

    def export_to_dict(self) -> Dict:
        """导出记忆系统状态"""
        return {
            "total_stored": self.total_stored,
            "total_retrieved": self.total_retrieved,
            "total_compressed": self.total_compressed,
            "working_memory_size": self.size_working,
            "long_term_memory_size": self.size_long_term,
            "vector_db_size": self.size_vector_db,
            "event_distribution": self.get_event_distribution(),
            "recent_memories": [m.to_dict() for m in self.retrieve_recent(5)],
        }

    def clear(self):
        """清空所有记忆（慎用）"""
        self.working_memory.clear()
        self.long_term_memory.clear()
        self.vector_store.clear()
        self.total_stored = 0
        self.total_retrieved = 0
        self._id_counter = 0
