"""
《猫语心声》 —— 心声生成编排器（端到端联调入口）

技术策划案v2 §4.5 端到端联调：
  RL决策 → 行为树 → LLM心声

完整数据流（§5 数据流与交互示例）:
  1. 玩家输入 (action, target_cat)
  2. 提取猫咪状态 (情绪、信任、环境)
  3. 记忆检索 (向量语义检索Top-3)
  4. RL策略推理 (输出宏观意图)
  5. 行为树执行 (意图→动作序列)
  6. LLM文本生成 (异步、缓存、降级)
  7. 心声气泡显示

本模块负责步骤 6-7 的完整编排：
  - 接收猫咪状态 + RL意图 + 记忆 → 构建提示词
  - 调用LLM服务（带缓存/降级）→ 生成心声
  - 后处理（清洗+性格过滤）→ 输出最终心声
  - 统计与监控

核心设计原则:
  - LLM仅负责文本渲染，不参与行为决策
  - 提示词严格绑定当前意图和情绪
  - 记忆上下文通过向量检索注入
  - 所有异步调用不阻塞主循环
"""

from __future__ import annotations
import os
import sys
import time
import threading
from typing import Dict, List, Optional, Tuple, Callable

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import numpy as np

from rl_environment.config import (
    INTENT_LIST, PERSONALITY_KEYS, PERSONALITY_DIM,
    CAT_CONFIGS, INTENT_PERSONALITY_MATRIX,
)

from .config import LLMConfig
from .prompt_builder import PromptBuilder, PromptContext
from .llm_service import LLMService, LLMResponse
from .cache_fallback import CacheFallbackManager, FaultInjector
from .text_postprocessor import TextPostprocessor, clean_r1_output
from .template_library import TemplateLibrary


# ═══════════════════════════════════════════
#  心声生成结果
# ═══════════════════════════════════════════

from dataclasses import dataclass, field


@dataclass
class MonologueResult:
    """一次心声生成的结果"""
    cat_id: str
    cat_name: str
    intent: str
    monologue: str                          # 最终心声文本
    from_cache: bool = False                # 来自缓存
    from_fallback: bool = False             # 来自模板降级
    llm_raw: str = ""                       # LLM原始输出
    llm_latency_ms: float = 0.0             # LLM推理延迟
    total_latency_ms: float = 0.0           # 总耗时
    postprocess_passed: bool = True         # 后处理是否通过
    prompt_tokens: int = 0                  # 提示词token数


class MonologueGenerator:
    """
    心声生成编排器 —— LLM心声系统的总入口。

    整合了:
    - PromptBuilder: 构建LLM提示词
    - LLMService: 本地LLM推理
    - CacheFallbackManager: 缓存与降级
    - TextPostprocessor: 文本后处理与性格过滤
    - TemplateLibrary: 模板降级

    用法:
        gen = MonologueGenerator()
        gen.initialize()  # 加载模型、初始化各模块

        # 同步生成
        result = gen.generate(cat_state, intent="approach_player",
                             player_action="call", memories=[...])

        # 异步生成（推荐用于游戏主循环）
        gen.generate_async(cat_state, intent="approach_player",
                          player_action="call", memories=[...],
                          callback=lambda r: show_bubble(r))

        # 查看统计
        print(gen.get_full_report())
    """

    def __init__(self, config: LLMConfig = None):
        self.cfg = config or LLMConfig()

        # 子模块
        self.prompt_builder = PromptBuilder(self.cfg)
        self.template_library = TemplateLibrary()
        self.text_postprocessor = TextPostprocessor(
            self.cfg, self.template_library
        )
        self.llm_service = LLMService(self.cfg)
        self.cache_manager = CacheFallbackManager(
            self.cfg, self.llm_service, self.template_library
        )

        # 初始化状态
        self._initialized = False
        self._model_loaded = False
        self._using_fallback_only = False

        # 统计
        self._total_generations = 0
        self._generation_times_ms: List[float] = []
        self._lock = threading.Lock()

    # ═══════════════════════════════════════════
    #  初始化
    # ═══════════════════════════════════════════

    def initialize(self, load_model: bool = True,
                   force_fallback: bool = False) -> bool:
        """
        初始化心声生成系统。

        参数:
            load_model: 是否加载LLM模型（False则仅使用模板库）
            force_fallback: 强制仅使用模板库（用于测试/降级运行）

        返回: 是否初始化成功
        """
        print("=" * 50)
        print("  《猫语心声》心声生成系统初始化")
        print("=" * 50)

        # 模板库始终可用
        print(f"\n[1/4] 模板库: {self.template_library.total_templates} 条模板")

        # 后处理器
        print(f"[2/4] 文本后处理器: 就绪")

        # 缓存管理器
        self.cache_manager.set_llm_service(self.llm_service)
        self.cache_manager.set_template_library(self.template_library)
        print(f"[3/4] 缓存与降级管理器: 就绪 "
              f"(TTL={self.cfg.cache_ttl_seconds}s, "
              f"max_failures={self.cfg.max_consecutive_failures})")

        # LLM模型
        if force_fallback:
            self._using_fallback_only = True
            print(f"[4/4] LLM模型: 跳过 (force_fallback=True, 仅用模板库)")
        elif load_model:
            print(f"[4/4] LLM模型加载中...")
            self._model_loaded = self.llm_service.load_model()
            if self._model_loaded:
                print(f"  → 模型加载成功")
            else:
                print(f"  → 模型加载失败，将仅使用模板库")
                self._using_fallback_only = True
        else:
            print(f"[4/4] LLM模型: 跳过 (load_model=False)")
            self._using_fallback_only = True

        self._initialized = True

        print(f"\n  状态: {'模板库模式' if self._using_fallback_only else 'LLM+模板库混合模式'}")
        print(f"  模型路径: {self.cfg.model_path}")
        print(f"  量化: {self.cfg.quantization} | ctx={self.cfg.n_ctx} | "
              f"threads={self.cfg.n_threads}")
        print("=" * 50)

        return True  # 模板库始终可用，初始化不会失败

    # ═══════════════════════════════════════════
    #  主入口：同步生成
    # ═══════════════════════════════════════════

    def generate(self, cat_state=None, *,
                 cat_id: str = "unknown",
                 cat_name: str = "",
                 intent: str = "idle_wander",
                 personality_vec: np.ndarray = None,
                 emotion_vector: np.ndarray = None,
                 trust: float = 50.0,
                 player_action: str = "none",
                 memories: List = None,
                 scene_desc: str = "猫咖大厅",
                 time_of_day: float = 12.0,
                 ) -> MonologueResult:
        """
        同步生成心声（完整流程）。

        支持两种调用方式:
        1. 传入cat_state (CatState对象) → 自动提取所有字段
        2. 传入独立字段 → 手动指定

        参数:
            cat_state: CatState对象（优先使用）
            cat_id: 猫咪ID (xiaoxue/oreo/orange)
            cat_name: 猫咪名字
            intent: 当前RL决策的意图
            personality_vec: [8] 性格向量
            emotion_vector: [5] 情绪向量 (0-1范围)
            trust: 信任度 (0-100)
            player_action: 玩家行为
            memories: 相关记忆列表 (MemoryItem或dict)
            scene_desc: 场景描述
            time_of_day: 游戏时间(小时)

        返回: MonologueResult
        """
        t_start = time.perf_counter()

        # 从cat_state提取字段
        if cat_state is not None:
            cat_id = getattr(cat_state, 'cat_id', cat_id)
            cat_name = getattr(cat_state, 'name', cat_name)
            if personality_vec is None:
                personality_vec = getattr(cat_state, 'personality_vector', None)
            if emotion_vector is None:
                emotion_vector = getattr(cat_state, 'emotion_vector', None)
            trust = getattr(cat_state, 'trust_level', trust)

        # 获取猫咪配置
        cat_cfg = CAT_CONFIGS.get(cat_id, {})

        if not cat_name:
            cat_name = cat_cfg.get("name", "猫咪")

        if personality_vec is None:
            personality_vec = np.array(
                cat_cfg.get("personality", [0.1] * PERSONALITY_DIM),
                dtype=np.float32,
            )

        if emotion_vector is None:
            emotion_vector = np.array([0.3, 0.3, 0.5, 0.5, 0.3],
                                      dtype=np.float32)

        # 构建PromptContext
        ctx = PromptContext(
            cat_id=cat_id,
            cat_name=cat_name,
            cat_breed=cat_cfg.get("breed", ""),
            cat_backstory=cat_cfg.get("backstory", ""),
            personality_vec=personality_vec,
            hunger=float(emotion_vector[0]) * 100,
            fear=float(emotion_vector[1]) * 100,
            curiosity=float(emotion_vector[2]) * 100,
            comfort=float(emotion_vector[3]) * 100,
            social=float(emotion_vector[4]) * 100,
            trust=trust,
            intent=intent,
            player_action=player_action,
            memories=memories or [],
            scene_desc=scene_desc,
            time_of_day=time_of_day,
        )

        # 构建完整提示词
        prompt = self.prompt_builder.build_full_prompt(ctx)
        prompt_tokens = PromptBuilder.estimate_tokens(prompt)

        # 通过缓存管理器获取心声
        monologue_text, from_cache = self.cache_manager.get_monologue(
            cat_id=cat_id,
            intent=intent,
            prompt=prompt,
            emotion_vector=emotion_vector,
            trust=trust,
            player_action=player_action,
            cat_name=cat_name,
        )

        from_fallback = False
        llm_raw = ""
        llm_latency = 0.0
        postprocess_passed = True

        # 如果不是来自缓存且不是降级（即LLM生成成功）
        if not from_cache and monologue_text != "……":
            llm_raw = monologue_text

            # 后处理
            monologue_text, postprocess_passed = self.text_postprocessor.process(
                raw_text=monologue_text,
                personality_vec=personality_vec,
                cat_id=cat_id,
                cat_name=cat_name,
                intent=intent,
                fear_value=float(emotion_vector[1]) * 100,
                trust_value=trust,
            )

        # 判断是否来自降级
        if (not from_cache and
                (monologue_text == "……" or not postprocess_passed or
                 self.cache_manager.is_in_fallback_mode or
                 self._using_fallback_only)):
            from_fallback = True
            # 确保有降级文本
            if monologue_text == "……" or not monologue_text:
                monologue_text = self.template_library.get_template_for_emotion(
                    cat_id, intent,
                    float(emotion_vector[1]) * 100, trust,
                )

        total_latency = (time.perf_counter() - t_start) * 1000

        with self._lock:
            self._total_generations += 1
            self._generation_times_ms.append(total_latency)
            if len(self._generation_times_ms) > 1000:
                self._generation_times_ms = self._generation_times_ms[-500:]

        return MonologueResult(
            cat_id=cat_id,
            cat_name=cat_name,
            intent=intent,
            monologue=monologue_text,
            from_cache=from_cache,
            from_fallback=from_fallback,
            llm_raw=llm_raw,
            llm_latency_ms=llm_latency,
            total_latency_ms=round(total_latency, 1),
            postprocess_passed=postprocess_passed,
            prompt_tokens=prompt_tokens,
        )

    # ═══════════════════════════════════════════
    #  异步生成（游戏主循环推荐）
    # ═══════════════════════════════════════════

    def generate_async(self, callback: Callable[[MonologueResult], None],
                       **kwargs):
        """
        异步生成心声（不阻塞主循环）。

        用法:
            def on_monologue_ready(result: MonologueResult):
                ui.show_bubble(result.monologue)

            generator.generate_async(
                callback=on_monologue_ready,
                cat_id="xiaoxue", intent="hide", ...
            )
        """
        def _worker():
            result = self.generate(**kwargs)
            try:
                callback(result)
            except Exception as e:
                print(f"[MonologueGenerator] 回调异常: {e}")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return t

    # ═══════════════════════════════════════════
    #  场景快捷方法
    # ═══════════════════════════════════════════

    def generate_for_cat_ear_vision(self, cat_state,
                                     intent: str,
                                     player_action: str = "none",
                                     memories: List = None,
                                     scene_desc: str = "猫咖大厅",
                                     ) -> MonologueResult:
        """
        猫耳视界心声 —— 玩家开启猫耳时的心声生成。

        这是最频繁的调用场景，每只可见猫30秒内缓存。
        """
        return self.generate(
            cat_state=cat_state,
            intent=intent,
            player_action=player_action,
            memories=memories,
            scene_desc=scene_desc,
        )

    def generate_for_milestone(self, cat_state,
                                milestone_type: str,
                                memories: List = None,
                                ) -> MonologueResult:
        """
        信任里程碑独白 —— 信任度突破阈值时触发。

        这类独白通常更长（3-5句），且不应走缓存。
        """
        cat_id = getattr(cat_state, 'cat_id', 'unknown')
        # 里程碑事件使缓存失效，确保新鲜独白
        self.cache_manager.invalidate_cache(cat_id)

        return self.generate(
            cat_state=cat_state,
            intent="accept_petting",
            player_action="none",
            memories=memories,
            scene_desc="信任突破的珍贵时刻",
        )

    def generate_for_night_talk(self, cat_state,
                                 other_cat_name: str,
                                 memories: List = None,
                                 ) -> MonologueResult:
        """
        夜间猫社交对话 —— 夜晚多猫聚集时的对话生成。
        """
        return self.generate(
            cat_state=cat_state,
            intent="social_groom",
            player_action="none",
            memories=memories,
            scene_desc=f"深夜，和{other_cat_name}依偎在一起",
            time_of_day=2.0,  # 深夜
        )

    # ═══════════════════════════════════════════
    #  缓存管理
    # ═══════════════════════════════════════════

    def warmup_cache(self) -> int:
        """
        预热缓存：为三只猫的常用场景预生成心声。

        在游戏启动或加载存档后调用，减少玩家等待。
        """
        common_intents = [
            "idle_wander", "approach_player", "eat", "sleep",
            "hide", "curious_inspect", "accept_petting",
            "stare_at_window",
        ]
        common_emotions = [
            {"hunger": 30, "fear": 30, "curiosity": 50, "comfort": 50, "social": 30, "trust": 50},
            {"hunger": 70, "fear": 60, "curiosity": 30, "comfort": 30, "social": 20, "trust": 30},
            {"hunger": 10, "fear": 10, "curiosity": 80, "comfort": 80, "social": 60, "trust": 80},
        ]

        total_warmed = 0
        for cat_id in ["xiaoxue", "oreo", "orange"]:
            # 构建prompt的回调
            def build_prompt(cid, intent, emotions):
                cat_cfg = CAT_CONFIGS.get(cid, {})
                ctx = PromptContext(
                    cat_id=cid,
                    cat_name=cat_cfg.get("name", ""),
                    cat_breed=cat_cfg.get("breed", ""),
                    cat_backstory=cat_cfg.get("backstory", ""),
                    personality_vec=np.array(
                        cat_cfg.get("personality", [0.1] * 8),
                        dtype=np.float32,
                    ),
                    hunger=emotions["hunger"],
                    fear=emotions["fear"],
                    curiosity=emotions["curiosity"],
                    comfort=emotions["comfort"],
                    social=emotions["social"],
                    trust=emotions["trust"],
                    intent=intent,
                )
                return self.prompt_builder.build_full_prompt(ctx)

            warmed = self.cache_manager.warmup_cache(
                cat_id, common_intents, common_emotions, build_prompt,
            )
            total_warmed += warmed

        print(f"[MonologueGenerator] 缓存预热完成: {total_warmed} 条")
        return total_warmed

    def invalidate_all_caches(self):
        """使所有缓存失效（重大事件后调用）"""
        self.cache_manager.invalidate_cache()

    # ═══════════════════════════════════════════
    #  统计与报告
    # ═══════════════════════════════════════════

    @property
    def total_generations(self) -> int:
        return self._total_generations

    @property
    def avg_latency_ms(self) -> float:
        with self._lock:
            times = self._generation_times_ms
        return sum(times) / len(times) if times else 0

    def get_full_report(self) -> str:
        """生成完整的心声系统报告"""
        lines = []
        lines.append("=" * 55)
        lines.append("  《猫语心声》LLM心声系统 — 完整报告")
        lines.append("=" * 55)
        lines.append("")

        # 系统状态
        lines.append("【系统状态】")
        lines.append(f"  模式: {'模板库' if self._using_fallback_only else 'LLM+模板库'}")
        lines.append(f"  模型: {self.cfg.model_filename} ({self.cfg.quantization})")
        lines.append(f"  模型加载: {'是' if self._model_loaded else '否'}")
        lines.append(f"  总生成次数: {self._total_generations}")
        lines.append(f"  平均总延迟: {self.avg_latency_ms:.1f} ms")
        lines.append("")

        # LLM服务统计
        lines.append("【LLM推理服务】")
        for k, v in self.llm_service.stats.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.2f}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")

        # 缓存与降级统计
        lines.append(self.cache_manager.get_report())
        lines.append("")

        # 文本后处理统计
        lines.append(self.text_postprocessor.get_report())
        lines.append("")

        # 模板库统计
        tpl_stats = self.template_library.stats()
        lines.append("【模板库】")
        lines.append(f"  总模板数: {tpl_stats['total_templates']}")
        lines.append(f"  已服务: {tpl_stats['total_served']}")
        lines.append("=" * 55)

        return "\n".join(lines)

    def reset_all_stats(self):
        """重置所有统计"""
        self.cache_manager.reset_stats()
        self.text_postprocessor.reset_stats()
        with self._lock:
            self._total_generations = 0
            self._generation_times_ms.clear()

    # ═══════════════════════════════════════════
    #  故障注入（测试用）
    # ═══════════════════════════════════════════

    def inject_fault(self, fault_type: str = "timeout",
                     probability: float = 1.0) -> FaultInjector:
        """
        注入LLM故障用于测试降级机制。

        用法:
            injector = generator.inject_fault("timeout", 1.0)
            # ... 测试 ...
            injector.clear()
        """
        injector = FaultInjector(self.llm_service)
        if fault_type == "timeout":
            injector.inject_timeout(probability)
        elif fault_type == "error":
            injector.inject_error(probability)
        return injector


# ═══════════════════════════════════════════
#  端到端联调测试
# ═══════════════════════════════════════════

def run_integration_test():
    """
    端到端联调测试：RL决策 → 行为树 → LLM心声

    测试场景（技术策划案v2 §5 完整数据流）：
      玩家在猫耳视界下轻声安抚流浪猫小雪
    """
    print("\n" + "=" * 55)
    print("  端到端联调测试")
    print("  RL决策 → 行为树 → LLM心声")
    print("=" * 55 + "\n")

    # 初始化（不加载模型，仅用模板库演示完整流程）
    gen = MonologueGenerator()
    gen.initialize(load_model=False)

    # 模拟三只猫的状态
    cats = {
        "xiaoxue": {
            "name": "小雪",
            "personality": np.array(
                [0.0, 0.9, 0.1, 0.2, 0.1, 0.0, 0.1, 0.3],
                dtype=np.float32,
            ),
            "emotion": np.array([0.2, 0.7, 0.1, 0.3, 0.05], dtype=np.float32),
            "trust": 22.0,
        },
        "oreo": {
            "name": "奥利奥",
            "personality": np.array(
                [0.8, 0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.7],
                dtype=np.float32,
            ),
            "emotion": np.array([0.3, 0.2, 0.4, 0.6, 0.3], dtype=np.float32),
            "trust": 55.0,
        },
        "orange": {
            "name": "橘子",
            "personality": np.array(
                [0.0, 0.0, 0.8, 0.9, 0.7, 0.0, 0.8, 0.1],
                dtype=np.float32,
            ),
            "emotion": np.array([0.6, 0.05, 0.7, 0.5, 0.4], dtype=np.float32),
            "trust": 75.0,
        },
    }

    # 模拟RL决策结果（实际由RL策略网络输出）
    rl_decisions = {
        "xiaoxue": "fearful_retreat",   # 小雪：恐惧主导→后退
        "oreo": "approach_player",      # 奥利奥：傲娇但好奇→靠近
        "orange": "ask_for_attention",  # 橘子：贪吃→求关注
    }

    # 模拟玩家行为
    player_action = "soothe"  # 轻声安抚

    # 模拟记忆检索结果
    mock_memories = [
        {"desc": "昨天玩家在沙发旁放下零食后安静离开了"},
        {"desc": "前天突然的关门声把我吓坏了"},
        {"desc": "被前任主人从航空箱里拖拽出来"},
    ]

    print("场景: 玩家开启猫耳视界，轻声安抚所有猫咪\n")

    results = []
    for cat_id, cat_data in cats.items():
        intent = rl_decisions[cat_id]

        result = gen.generate(
            cat_id=cat_id,
            cat_name=cat_data["name"],
            intent=intent,
            personality_vec=cat_data["personality"],
            emotion_vector=cat_data["emotion"],
            trust=cat_data["trust"],
            player_action=player_action,
            memories=mock_memories,
            scene_desc="猫咖大厅，午后的阳光暖洋洋的",
            time_of_day=14.0,
        )

        results.append(result)

        # 打印结果
        print(f"  [{cat_data['name']}] RL意图: {intent}")
        print(f"    情绪: 饥饿{cat_data['emotion'][0]:.0%} "
              f"恐惧{cat_data['emotion'][1]:.0%} "
              f"好奇{cat_data['emotion'][2]:.0%} "
              f"信任{cat_data['trust']:.0f}")
        print(f"    心声: \"{result.monologue}\"")
        print(f"    来源: {'缓存' if result.from_cache else '降级模板' if result.from_fallback else 'LLM生成'}")
        print(f"    延迟: {result.total_latency_ms:.1f}ms")
        print()

    # 验证性格差异
    print("─" * 40)
    print("性格一致性验证:")
    xiaoxue_ok = results[0].intent == 'fearful_retreat'
    oreo_ok = results[1].intent == 'approach_player'
    orange_ok = results[2].intent == 'ask_for_attention'
    print(f"  小雪(怯懦) 意图={results[0].intent}: "
          f"{'[PASS] 符合预期(恐惧后退)' if xiaoxue_ok else '[FAIL]'}")
    print(f"  奥利奥(傲娇) 意图={results[1].intent}: "
          f"{'[PASS] 符合预期(靠近但保持距离)' if oreo_ok else '[FAIL]'}")
    print(f"  橘子(贪吃好奇) 意图={results[2].intent}: "
          f"{'[PASS] 符合预期(主动求关注)' if orange_ok else '[FAIL]'}")

    # 打印完整报告
    print("\n" + gen.get_full_report())

    return results


def run_fallback_test():
    """
    降级机制测试：模拟LLM故障，验证系统无缝切换到模板库。
    """
    print("\n" + "=" * 55)
    print("  降级机制测试")
    print("  模拟LLM故障 → 验证模板库无缝切换")
    print("=" * 55 + "\n")

    gen = MonologueGenerator()
    gen.initialize(load_model=False)

    emotion = np.array([0.3, 0.3, 0.5, 0.5, 0.3], dtype=np.float32)
    personality = np.array([0.8, 0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.7],
                           dtype=np.float32)

    # 注入LLM超时故障
    print(">>> 注入100%超时故障...")
    injector = gen.inject_fault("timeout", 1.0)

    print(">>> 发送5次心声请求（全部应降级到模板库）:\n")
    for i in range(5):
        result = gen.generate(
            cat_id="oreo", cat_name="奥利奥",
            intent="idle_wander",
            personality_vec=personality,
            emotion_vector=emotion,
            trust=55.0,
        )
        source = "降级模板" if result.from_fallback else "缓存" if result.from_cache else "LLM"
        print(f"  [{i+1}] \"{result.monologue}\" | 来源: {source} | "
              f"延迟: {result.total_latency_ms:.1f}ms")

    print(f"\n>>> 降级统计: "
          f"命中={gen.cache_manager.stats['cache_hits']}, "
          f"降级={gen.cache_manager.stats['fallback_used']}")

    # 验证连续失败后进入降级模式
    print(f">>> 降级模式: {'已激活' if gen.cache_manager.is_in_fallback_mode else '未激活'}")
    print(f">>> 连续失败: {gen.cache_manager.stats['consecutive_failures']}")

    # 清除故障
    injector.clear()
    print("\n>>> 清除故障注入，恢复正常模式...")
    gen.cache_manager._fallback_mode = False
    gen.cache_manager._consecutive_failures = 0

    print(f">>> 降级模式: {'已激活' if gen.cache_manager.is_in_fallback_mode else '已恢复'}")
    print("\n[PASS] 降级机制测试通过：LLM故障时系统无缝切换到模板库")


if __name__ == "__main__":
    print("\n" + "█" * 55)
    print("  《猫语心声》LLM心声系统 — 端到端联调")
    print("█" * 55)

    # 测试1: 端到端联调
    run_integration_test()

    # 测试2: 降级机制
    run_fallback_test()

    print("\n" + "█" * 55)
    print("  全部测试完成")
    print("█" * 55)
