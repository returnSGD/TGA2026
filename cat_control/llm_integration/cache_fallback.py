"""
《猫语心声》 —— 缓存与降级机制

技术策划案v2 §4.5.3 的完整实现：
- LRU缓存: 相同(cat_id + intent + emotion_bucket) 30秒内不重复生成
- 降级策略: 连续失败N次 → 冷却期 → 强制使用模板库
- 健康检查: 定期探测LLM服务，自动恢复
- 三层回退:
  1. LLM缓存命中 → 直接返回
  2. LLM推理成功 → 返回+写缓存
  3. LLM不可用 → 降级到TemplateLibrary模板库

设计原则（§4.4.3）:
  检查失败时，使用策划撰写的性格模板文本库作为fallback
"""

from __future__ import annotations
import time
import hashlib
import threading
from typing import Dict, Optional, Tuple, Callable
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .config import (
    LLMConfig,
    CACHE_TTL_SECONDS, CACHE_MAX_SIZE, EMOTION_BUCKET_SIZE,
    FALLBACK_COOLDOWN_SECONDS, MAX_CONSECUTIVE_FAILURES,
    LLM_HEALTH_CHECK_INTERVAL,
)


def _emotion_bucket(val: float) -> int:
    """情绪值 → 分桶索引 (0-4)"""
    val = max(0.0, min(99.9, val))
    return int(val // EMOTION_BUCKET_SIZE)


def _make_cache_key(cat_id: str, intent: str,
                    hunger: float, fear: float, curiosity: float,
                    comfort: float, social: float,
                    trust: float, player_action: str) -> str:
    """
    生成缓存键。

    键 = hash(cat_id|intent|hunger_bucket|fear_bucket|curiosity_bucket|
              comfort_bucket|social_bucket|trust_bucket|player_action)
    """
    parts = [
        cat_id,
        intent,
        str(_emotion_bucket(hunger)),
        str(_emotion_bucket(fear)),
        str(_emotion_bucket(curiosity)),
        str(_emotion_bucket(comfort)),
        str(_emotion_bucket(social)),
        str(_emotion_bucket(trust)),
        player_action,
    ]
    raw = "|".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()


@dataclass
class CacheEntry:
    """缓存条目"""
    text: str
    created_at: float
    access_count: int = 0


@dataclass
class FallbackStats:
    """降级统计"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    llm_successes: int = 0
    llm_timeouts: int = 0
    llm_errors: int = 0
    fallback_used: int = 0
    fallback_cooldown_triggered: int = 0


class LRUCache:
    """线程安全的LRU缓存"""

    def __init__(self, max_size: int = CACHE_MAX_SIZE,
                 ttl_seconds: float = CACHE_TTL_SECONDS):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        """获取缓存，返回None表示未命中或已过期"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if time.time() - entry.created_at > self._ttl:
                del self._cache[key]
                return None
            # 移到末尾（最近使用）
            self._cache.move_to_end(key)
            entry.access_count += 1
            return entry.text

    def set(self, key: str, text: str):
        """写入缓存"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = CacheEntry(
                    text=text, created_at=time.time(),
                    access_count=self._cache[key].access_count + 1,
                )
                return

            if len(self._cache) >= self._max_size:
                # 淘汰最久未使用
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(text=text, created_at=time.time())

    def invalidate(self, cat_id: str = None):
        """失效缓存（可在重大事件后调用）"""
        with self._lock:
            if cat_id is None:
                self._cache.clear()
            else:
                keys_to_remove = [
                    k for k in self._cache if k.startswith(cat_id)
                ]
                for k in keys_to_remove:
                    del self._cache[k]

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def cleanup_expired(self) -> int:
        """清理过期条目，返回清理数量"""
        now = time.time()
        removed = 0
        with self._lock:
            keys_to_remove = [
                k for k, v in self._cache.items()
                if now - v.created_at > self._ttl * 2
            ]
            for k in keys_to_remove:
                del self._cache[k]
                removed += 1
        return removed


class CacheFallbackManager:
    """
    缓存与降级管理器。

    协调 LLM推理 ←→ 缓存 ←→ 模板库 三条路径：

    游戏主循环调用顺序:
    1. try_cache() → 命中则直接返回
    2. try_llm() → LLM生成成功则写缓存返回
    3. fallback_to_template() → LLM不可用时走模板库

    降级状态机:
      NORMAL ──连续失败≥3──→ COOLDOWN(5秒) ──时间到──→ NORMAL
      COOLDOWN期间所有请求直接走模板库
    """

    def __init__(self, config: LLMConfig = None,
                 llm_service=None, template_library=None):
        self.cfg = config or LLMConfig()
        self._llm = llm_service
        self._templates = template_library

        # LRU缓存
        self._cache = LRUCache(
            max_size=self.cfg.cache_max_size,
            ttl_seconds=self.cfg.cache_ttl_seconds,
        )

        # 降级状态
        self._fallback_mode = False
        self._fallback_until = 0.0
        self._consecutive_failures = 0
        self._last_health_check = 0.0
        self._lock = threading.Lock()

        # 统计
        self._stats = FallbackStats()

    def set_llm_service(self, llm_service):
        self._llm = llm_service

    def set_template_library(self, template_library):
        self._templates = template_library

    # ═══════════════════════════════════════════
    #  主入口
    # ═══════════════════════════════════════════

    def get_monologue(self, cat_id: str, intent: str,
                      prompt: str,
                      emotion_vector: np.ndarray = None,
                      trust: float = 50.0,
                      player_action: str = "none",
                      cat_name: str = "",
                      ) -> Tuple[str, bool]:
        """
        获取心声文本（缓存优先 → LLM推理 → 模板降级）。

        参数:
            cat_id: 猫咪ID
            intent: 当前意图
            prompt: 完整LLM提示词
            emotion_vector: [5] 情绪向量 (可选，用于缓存键)
            trust: 信任度
            player_action: 玩家行为
            cat_name: 猫咪名字（降级时给模板库）

        返回: (心声文本, 是否来自缓存)
        """
        self._stats.total_requests += 1

        # 提取情绪值
        if emotion_vector is not None and len(emotion_vector) >= 5:
            hunger, fear, cur, comf, soc = emotion_vector[:5]
            hunger, fear, cur = hunger * 100, fear * 100, cur * 100
            comf, soc = comf * 100, soc * 100
        else:
            hunger = fear = cur = comf = soc = 30.0

        # 1. 尝试缓存
        cache_key = _make_cache_key(
            cat_id, intent, hunger, fear, cur, comf, soc, trust, player_action,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._stats.cache_hits += 1
            return cached, True

        self._stats.cache_misses += 1

        # 2. 检查降级状态
        if self._should_skip_llm():
            self._stats.fallback_used += 1
            fallback_text = self._get_fallback(cat_id, intent, cat_name,
                                               fear, trust)
            return fallback_text, False

        # 3. LLM推理
        if self._llm is not None:
            response = self._llm.generate(prompt)
            if response.finish_reason == "stop":
                self._stats.llm_successes += 1
                self._on_llm_success()
                self._cache.set(cache_key, response.text)
                return response.text, False
            elif response.finish_reason == "timeout":
                self._stats.llm_timeouts += 1
                self._on_llm_failure()
            else:
                self._stats.llm_errors += 1
                self._on_llm_failure()

        # 4. 降级到模板库
        self._stats.fallback_used += 1
        fallback_text = self._get_fallback(cat_id, intent, cat_name,
                                           fear, trust)
        return fallback_text, False

    # ═══════════════════════════════════════════
    #  降级状态机
    # ═══════════════════════════════════════════

    def _should_skip_llm(self) -> bool:
        """判断是否应跳过LLM直接用模板"""
        with self._lock:
            if self._fallback_mode:
                if time.time() >= self._fallback_until:
                    self._fallback_mode = False
                    self._consecutive_failures = 0
                    self._stats.fallback_cooldown_triggered += 1
                    return False
                return True
            return False

    def _on_llm_success(self):
        """LLM调用成功，重置失败计数"""
        with self._lock:
            self._consecutive_failures = 0
            if self._fallback_mode and time.time() >= self._fallback_until:
                self._fallback_mode = False

    def _on_llm_failure(self):
        """LLM调用失败，累加计数，达到阈值进入降级模式"""
        with self._lock:
            self._consecutive_failures += 1
            if (self._consecutive_failures >=
                    self.cfg.max_consecutive_failures):
                self._fallback_mode = True
                self._fallback_until = (
                    time.time() + self.cfg.fallback_cooldown_seconds
                )

    def _get_fallback(self, cat_id: str, intent: str, cat_name: str,
                      fear: float = 30.0, trust: float = 50.0) -> str:
        """从模板库获取降级文本"""
        if self._templates is not None:
            return self._templates.get_template_for_emotion(
                cat_id, intent, fear, trust
            )
        return "……"

    # ═══════════════════════════════════════════
    #  健康检查
    # ═══════════════════════════════════════════

    def health_check(self) -> bool:
        """检查LLM服务健康状态，决定是否可恢复正常模式"""
        now = time.time()
        if now - self._last_health_check < self.cfg.health_check_interval:
            return not self._fallback_mode

        self._last_health_check = now

        if self._llm is not None and self._llm.is_healthy:
            with self._lock:
                if self._fallback_mode and now >= self._fallback_until:
                    self._fallback_mode = False
                    self._consecutive_failures = 0
                    return True
        return not self._fallback_mode

    # ═══════════════════════════════════════════
    #  缓存管理
    # ═══════════════════════════════════════════

    def invalidate_cache(self, cat_id: str = None):
        """使缓存失效（重大事件后调用，确保心声新鲜度）"""
        self._cache.invalidate(cat_id)

    def warmup_cache(self, cat_id: str, intents: list,
                     emotion_combos: list, prompt_builder_fn: Callable):
        """预热缓存：为常用场景预生成心声"""
        warmed = 0
        for intent in intents:
            for emotions in emotion_combos:
                prompt = prompt_builder_fn(cat_id, intent, emotions)
                if self._llm is not None:
                    response = self._llm.generate(prompt)
                    if response.finish_reason == "stop":
                        cache_key = _make_cache_key(
                            cat_id, intent,
                            emotions.get("hunger", 30),
                            emotions.get("fear", 30),
                            emotions.get("curiosity", 50),
                            emotions.get("comfort", 50),
                            emotions.get("social", 30),
                            emotions.get("trust", 50),
                            "none",
                        )
                        self._cache.set(cache_key, response.text)
                        warmed += 1
        return warmed

    # ═══════════════════════════════════════════
    #  统计
    # ═══════════════════════════════════════════

    @property
    def cache_hit_rate(self) -> float:
        total = self._stats.cache_hits + self._stats.cache_misses
        return self._stats.cache_hits / max(1, total)

    @property
    def fallback_rate(self) -> float:
        return self._stats.fallback_used / max(1, self._stats.total_requests)

    @property
    def is_in_fallback_mode(self) -> bool:
        return self._fallback_mode

    @property
    def stats(self) -> Dict:
        return {
            **self._stats.__dict__,
            "cache_size": self._cache.size,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "fallback_rate": round(self.fallback_rate, 4),
            "in_fallback_mode": self._fallback_mode,
            "consecutive_failures": self._consecutive_failures,
            "cache_ttl_seconds": self.cfg.cache_ttl_seconds,
            "fallback_cooldown_seconds": self.cfg.fallback_cooldown_seconds,
        }

    def get_report(self) -> str:
        """生成可读的统计报告"""
        s = self._stats
        return (
            f"══════════ 缓存与降级报告 ══════════\n"
            f"  总请求数:        {s.total_requests}\n"
            f"  缓存命中:        {s.cache_hits} "
            f"({self.cache_hit_rate:.1%})\n"
            f"  缓存未命中:      {s.cache_misses}\n"
            f"  LLM成功:         {s.llm_successes}\n"
            f"  LLM超时:         {s.llm_timeouts}\n"
            f"  LLM错误:         {s.llm_errors}\n"
            f"  降级使用:        {s.fallback_used} "
            f"({self.fallback_rate:.1%})\n"
            f"  冷却触发次数:    {s.fallback_cooldown_triggered}\n"
            f"  当前缓存条目:    {self._cache.size}\n"
            f"  降级模式:        {'是' if self._fallback_mode else '否'}\n"
            f"  连续失败:        {self._consecutive_failures}\n"
            f"══════════════════════════════════════"
        )

    def reset_stats(self):
        """重置统计"""
        self._stats = FallbackStats()
        self._consecutive_failures = 0


# ═══════════════════════════════════════════
#  测试辅助：模拟LLM故障
# ═══════════════════════════════════════════

class FaultInjector:
    """
    故障注入器 —— 用于测试降级机制。

    用法:
      injector = FaultInjector(llm_service)
      injector.inject_timeout(probability=1.0)  # 100%超时
      # ... 运行测试 ...
      injector.clear()
    """

    def __init__(self, llm_service):
        self._llm = llm_service
        self._original_generate = llm_service.generate
        self._fault_type = None
        self._fault_probability = 0.0

    def inject_timeout(self, probability: float = 1.0):
        """注入超时故障"""
        self._fault_type = "timeout"
        self._fault_probability = probability
        self._install()

    def inject_error(self, probability: float = 1.0):
        """注入错误故障"""
        self._fault_type = "error"
        self._fault_probability = probability
        self._install()

    def clear(self):
        """清除故障注入"""
        self._llm.generate = self._original_generate
        self._fault_type = None
        self._fault_probability = 0.0

    def _install(self):
        import random
        original = self._original_generate
        fault_type = self._fault_type
        fault_prob = self._fault_probability

        from .llm_service import LLMResponse

        def _faulty_generate(prompt, **kwargs):
            if random.random() < fault_prob:
                if fault_type == "timeout":
                    return LLMResponse(
                        text="", tokens_generated=0, latency_ms=999,
                        finish_reason="timeout",
                    )
                else:
                    return LLMResponse(
                        text="", tokens_generated=0, latency_ms=50,
                        finish_reason="error:injected_fault",
                    )
            return original(prompt, **kwargs)

        self._llm.generate = _faulty_generate
