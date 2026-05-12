"""
《猫语心声》 —— 向量数据库抽象层

支持三种后端：
- numpy: 纯NumPy内存存储（默认，零依赖，500条规模下<1ms检索）
- sqlite_vec: sqlite-vec扩展（持久化，支持大规模）
- chroma: Chroma向量数据库（功能最全，需额外安装）

统一接口: add / search / delete / count
"""

from __future__ import annotations
import os
import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VectorEntry:
    """向量条目"""
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class VectorStore(ABC):
    """向量数据库抽象基类"""

    @abstractmethod
    def add(self, entry_id: str, vector: np.ndarray,
            metadata: Dict[str, Any] = None) -> bool:
        """添加或更新向量"""
        ...

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int = 3
              ) -> List[Tuple[str, float, Dict]]:
        """余弦相似度检索，返回 [(id, similarity, metadata), ...]"""
        ...

    @abstractmethod
    def delete(self, entry_id: str) -> bool:
        """删除向量"""
        ...

    @abstractmethod
    def count(self) -> int:
        """当前向量数量"""
        ...

    @abstractmethod
    def clear(self):
        """清空所有向量"""
        ...

    def batch_add(self, entries: List[VectorEntry]) -> int:
        """批量添加，返回成功数量"""
        count = 0
        for entry in entries:
            if self.add(entry.id, entry.vector, entry.metadata):
                count += 1
        return count


class NumpyVectorStore(VectorStore):
    """
    纯NumPy内存向量存储。

    500条 × 128维 → <1ms 余弦相似度检索。
    适用场景：本地开发、单元测试、嵌入式部署。
    """

    def __init__(self, embed_dim: int = 128):
        self.embed_dim = embed_dim
        self._ids: List[str] = []
        self._vectors: List[np.ndarray] = []
        self._metadatas: List[Dict] = []
        self._id_to_idx: Dict[str, int] = {}

    def add(self, entry_id: str, vector: np.ndarray,
            metadata: Dict[str, Any] = None) -> bool:
        vector = np.asarray(vector, dtype=np.float32).flatten()
        if vector.shape[0] != self.embed_dim:
            vector = self._pad_or_truncate(vector)

        # 归一化存储
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        if entry_id in self._id_to_idx:
            idx = self._id_to_idx[entry_id]
            self._vectors[idx] = vector
            self._metadatas[idx] = metadata or {}
            return True

        self._id_to_idx[entry_id] = len(self._ids)
        self._ids.append(entry_id)
        self._vectors.append(vector)
        self._metadatas.append(metadata or {})
        return True

    def search(self, query: np.ndarray, top_k: int = 3
              ) -> List[Tuple[str, float, Dict]]:
        if not self._vectors:
            return []

        query = np.asarray(query, dtype=np.float32).flatten()
        if query.shape[0] != self.embed_dim:
            query = self._pad_or_truncate(query)

        q_norm = np.linalg.norm(query)
        if q_norm > 0:
            query = query / q_norm

        # 批量余弦相似度
        vecs = np.stack(self._vectors, axis=0)  # [N, D]
        sims = np.dot(vecs, query)               # [N]

        # Top-K
        if len(sims) <= top_k:
            indices = np.argsort(-sims)
        else:
            indices = np.argpartition(-sims, top_k)[:top_k]
            indices = indices[np.argsort(-sims[indices])]

        results = []
        for idx in indices:
            results.append((
                self._ids[idx],
                float(sims[idx]),
                dict(self._metadatas[idx]),
            ))
        return results

    def delete(self, entry_id: str) -> bool:
        if entry_id not in self._id_to_idx:
            return False
        idx = self._id_to_idx.pop(entry_id)
        self._ids.pop(idx)
        self._vectors.pop(idx)
        self._metadatas.pop(idx)
        # 重建索引映射
        self._id_to_idx = {eid: i for i, eid in enumerate(self._ids)}
        return True

    def count(self) -> int:
        return len(self._ids)

    def clear(self):
        self._ids.clear()
        self._vectors.clear()
        self._metadatas.clear()
        self._id_to_idx.clear()

    def _pad_or_truncate(self, vec: np.ndarray) -> np.ndarray:
        """填充或截断到目标维度"""
        if len(vec) > self.embed_dim:
            return vec[:self.embed_dim]
        padded = np.zeros(self.embed_dim, dtype=np.float32)
        padded[:len(vec)] = vec
        return padded

    def get_all_ids(self) -> List[str]:
        return list(self._ids)

    def get_metadata(self, entry_id: str) -> Optional[Dict]:
        idx = self._id_to_idx.get(entry_id)
        if idx is not None:
            return dict(self._metadatas[idx])
        return None


# ─── sqlite-vec 后端（可选，需 pip install sqlite-vec） ───

class SQLiteVecStore(VectorStore):
    """
    sqlite-vec 向量存储（持久化到本地SQLite文件）。

    特点: 零服务依赖、支持过滤、ACID事务。
    pip install sqlite-vec
    """

    def __init__(self, db_path: str, embed_dim: int = 128,
                 table_name: str = "memory_vectors"):
        self.db_path = db_path
        self.embed_dim = embed_dim
        self.table_name = table_name
        self._conn = None
        self._init_db()

    def _init_db(self):
        try:
            import sqlite_vec
            import sqlite3
            self._conn = sqlite3.connect(self.db_path)
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)

            self._conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_name}
                USING vec0(
                    id TEXT PRIMARY KEY,
                    embedding FLOAT[{self.embed_dim}],
                    +metadata TEXT
                )
            """)
            self._conn.commit()
        except ImportError:
            raise ImportError(
                "sqlite-vec 未安装。请运行: pip install sqlite-vec\n"
                "或使用 NumpyVectorStore 作为后备。"
            )

    def add(self, entry_id: str, vector: np.ndarray,
            metadata: Dict[str, Any] = None) -> bool:
        import json
        vector = np.asarray(vector, dtype=np.float32).flatten()
        if vector.shape[0] != self.embed_dim:
            return False

        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        try:
            self._conn.execute(
                f"INSERT OR REPLACE INTO {self.table_name} (id, embedding, metadata) "
                f"VALUES (?, ?, ?)",
                (entry_id, vector.tobytes(), meta_json)
            )
            self._conn.commit()
            return True
        except Exception:
            return False

    def search(self, query: np.ndarray, top_k: int = 3
              ) -> List[Tuple[str, float, Dict]]:
        import json
        query = np.asarray(query, dtype=np.float32).flatten()
        try:
            rows = self._conn.execute(
                f"SELECT id, distance, metadata FROM {self.table_name} "
                f"WHERE embedding MATCH ? AND k = ? "
                f"ORDER BY distance",
                (query.tobytes(), top_k)
            ).fetchall()

            results = []
            for row in rows:
                entry_id, distance, meta_json = row
                similarity = 1.0 - float(distance)
                metadata = json.loads(meta_json) if meta_json else {}
                results.append((entry_id, similarity, metadata))
            return results
        except Exception:
            return []

    def delete(self, entry_id: str) -> bool:
        try:
            self._conn.execute(
                f"DELETE FROM {self.table_name} WHERE id = ?", (entry_id,)
            )
            self._conn.commit()
            return True
        except Exception:
            return False

    def count(self) -> int:
        try:
            row = self._conn.execute(
                f"SELECT COUNT(*) FROM {self.table_name}"
            ).fetchone()
            return row[0] if row else 0
        except Exception:
            return 0

    def clear(self):
        try:
            self._conn.execute(f"DELETE FROM {self.table_name}")
            self._conn.commit()
        except Exception:
            pass


# ─── Chroma 后端（可选，需 pip install chromadb） ───

class ChromaVectorStore(VectorStore):
    """
    Chroma 向量数据库（功能最全，支持持久化、元数据过滤）。

    pip install chromadb
    """

    def __init__(self, persist_dir: str, embed_dim: int = 128,
                 collection_name: str = "cat_memories"):
        self.persist_dir = persist_dir
        self.embed_dim = embed_dim
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._init()

    def _init(self):
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            try:
                self._collection = self._client.get_collection(self.collection_name)
            except Exception:
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
        except ImportError:
            raise ImportError(
                "chromadb 未安装。请运行: pip install chromadb\n"
                "或使用 NumpyVectorStore 作为后备。"
            )

    def add(self, entry_id: str, vector: np.ndarray,
            metadata: Dict[str, Any] = None) -> bool:
        vector = np.asarray(vector, dtype=np.float32).flatten()
        if vector.shape[0] != self.embed_dim:
            return False

        # 转换metadata值全为字符串（Chroma要求）
        str_meta = {k: str(v)[:512] for k, v in (metadata or {}).items()}

        try:
            self._collection.upsert(
                ids=[entry_id],
                embeddings=[vector.tolist()],
                metadatas=[str_meta],
            )
            return True
        except Exception:
            return False

    def search(self, query: np.ndarray, top_k: int = 3
              ) -> List[Tuple[str, float, Dict]]:
        query = np.asarray(query, dtype=np.float32).flatten()
        try:
            result = self._collection.query(
                query_embeddings=[query.tolist()],
                n_results=top_k,
            )
            results = []
            if result["ids"] and result["ids"][0]:
                for i, eid in enumerate(result["ids"][0]):
                    dist = result["distances"][0][i] if result["distances"] else 0
                    sim = 1.0 - float(dist)
                    meta = result["metadatas"][0][i] if result["metadatas"] else {}
                    results.append((eid, sim, meta))
            return results
        except Exception:
            return []

    def delete(self, entry_id: str) -> bool:
        try:
            self._collection.delete(ids=[entry_id])
            return True
        except Exception:
            return False

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            return 0

    def clear(self):
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            pass


def create_vector_store(backend: str = "numpy", **kwargs) -> VectorStore:
    """工厂函数：根据配置创建向量数据库实例"""
    embed_dim = kwargs.pop("embed_dim", 128)

    if backend == "sqlite_vec":
        db_path = kwargs.pop("db_path", "memory_vectors.db")
        return SQLiteVecStore(db_path=db_path, embed_dim=embed_dim, **kwargs)
    elif backend == "chroma":
        persist_dir = kwargs.pop("persist_dir", "./chroma_data")
        return ChromaVectorStore(persist_dir=persist_dir, embed_dim=embed_dim, **kwargs)
    else:
        return NumpyVectorStore(embed_dim=embed_dim)
