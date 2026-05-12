"""
《猫语心声》 —— 语义嵌入服务

使用 Sentence Transformer 将记忆文本编码为128维语义向量。
支持：
- all-MiniLM-L6-v2（默认，384维→128维投影，CPU <10ms/条）
- 本地缓存（相同文本不重复编码）
- numpy fallback（当模型不可用时使用随机投影）
"""

from __future__ import annotations
import os
import hashlib
import numpy as np
from typing import List, Optional, Dict
from functools import lru_cache


class EmbeddingService:
    """
    语义嵌入服务 —— 将自然语言记忆描述转为向量。

    模型: all-MiniLM-L6-v2 (384维原始输出 → 128维PCA/随机投影)
    延迟: <10ms/条 (CPU)
    缓存: LRU 最近1024条
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 target_dim: int = 128, device: str = "cpu",
                 cache_size: int = 1024):
        self.model_name = model_name
        self.target_dim = target_dim
        self.device = device
        self._model = None
        self._projection: Optional[np.ndarray] = None  # [384, 128] 投影矩阵
        self._model_loaded = False
        self._fallback = False
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def is_available(self) -> bool:
        return self._model_loaded and not self._fallback

    def load(self) -> bool:
        """加载Sentence Transformer模型。成功返回True。"""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model_loaded = True

            # 创建投影矩阵：384 → 128
            orig_dim = self._model.get_sentence_embedding_dimension()
            if orig_dim != self.target_dim:
                rng = np.random.RandomState(42)
                proj = rng.randn(orig_dim, self.target_dim).astype(np.float32)
                # 正交化
                u, _, vh = np.linalg.svd(proj, full_matrices=False)
                self._projection = (u @ vh).astype(np.float32)
                print(f"[Embedding] 模型 {self.model_name} 已加载 "
                      f"({orig_dim}→{self.target_dim}维投影)")
            else:
                self._projection = None
                print(f"[Embedding] 模型 {self.model_name} 已加载 ({orig_dim}维)")

            return True

        except ImportError:
            print("[Embedding] sentence-transformers 未安装，使用随机投影fallback")
            self._fallback = True
            self._model_loaded = True
            return False
        except Exception as e:
            print(f"[Embedding] 模型加载失败: {e}，使用随机投影fallback")
            self._fallback = True
            self._model_loaded = True
            return False

    def encode(self, text: str) -> np.ndarray:
        """编码单条文本为128维向量"""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key].copy()

        self._cache_misses += 1
        if self._fallback or not self._model:
            vec = self._encode_fallback(text)
        else:
            vec = self._encode_model(text)

        # 缓存
        if len(self._cache) > 1024:
            # 简单FIFO淘汰
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = vec.copy()
        return vec

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """批量编码"""
        vectors = []
        for text in texts:
            vectors.append(self.encode(text))
        return np.stack(vectors, axis=0)

    def _encode_model(self, text: str) -> np.ndarray:
        """使用Sentence Transformer编码"""
        emb = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0].astype(np.float32)

        if self._projection is not None:
            emb = emb @ self._projection
            emb = emb / (np.linalg.norm(emb) + 1e-8)

        # 确保目标维度
        if len(emb) > self.target_dim:
            emb = emb[:self.target_dim]
        elif len(emb) < self.target_dim:
            padded = np.zeros(self.target_dim, dtype=np.float32)
            padded[:len(emb)] = emb
            emb = padded

        return emb

    def _encode_fallback(self, text: str) -> np.ndarray:
        """
        随机投影fallback：基于文本的确定性哈希生成伪嵌入。

        虽然不是语义嵌入，但相同文本产生相同向量，
        仍可用于记忆匹配（相同事件召回相同记忆）。
        """
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:8], 'big'))
        vec = rng.randn(self.target_dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)

    @property
    def cache_hit_rate(self) -> float:
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / total if total > 0 else 0.0

    def stats(self) -> Dict:
        return {
            "model": self.model_name,
            "target_dim": self.target_dim,
            "model_loaded": self._model_loaded,
            "fallback_mode": self._fallback,
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": f"{self.cache_hit_rate:.1%}",
        }


# ─── 全局单例 ───

_global_embed_service: Optional[EmbeddingService] = None


def get_embedding_service(model_name: str = "all-MiniLM-L6-v2",
                         target_dim: int = 128,
                         device: str = "cpu") -> EmbeddingService:
    """获取全局嵌入服务单例"""
    global _global_embed_service
    if _global_embed_service is None:
        _global_embed_service = EmbeddingService(
            model_name=model_name, target_dim=target_dim, device=device
        )
        _global_embed_service.load()
    return _global_embed_service
