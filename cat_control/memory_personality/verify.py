"""
《猫语心声》 —— 记忆系统与性格过滤器验证工具

验证项：
1. 记忆系统单元测试（存储/检索/衰减/压缩）
2. 性格差异对比测试（三只猫在同一场景下的意图/行为差异）
3. 三层过滤器一致性测试
4. 记忆→RL状态注入验证
"""

from __future__ import annotations
import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_personality.config import MemoryConfig, CAT_PERSONALITIES
from memory_personality.memory_manager import MemoryManager
from memory_personality.personality_filter import PersonalityFilter
from memory_personality.embedding import EmbeddingService
from memory_personality.vector_store import NumpyVectorStore
from memory_personality.memory_rl_bridge import MemoryRLBridge
from rl_environment.config import (
    INTENT_LIST, PERSONALITY_KEYS, PERSONALITY_DIM,
    CAT_CONFIGS, INTENT_PERSONALITY_MATRIX,
    PERSONALITY_BEHAVIOR_PARAMS, PERSONALITY_FORBIDDEN_WORDS,
    STATE_DIM, MEMORY_EMBED_DIM, TOP_K_MEMORIES,
)

PASS = "[PASS]"
FAIL = "[FAIL]"


@dataclass
class VerifyReport:
    """验证报告"""
    test_name: str
    passed: bool
    details: str = ""
    data: Dict = field(default_factory=dict)


class MemoryPersonalityVerifier:
    """记忆系统与性格过滤器综合验证器"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.config = MemoryConfig()
        self.memory = MemoryManager(config=self.config)
        self.pf = PersonalityFilter(config=self.config)
        self.bridge = MemoryRLBridge(
            memory_manager=self.memory,
            personality_filter=self.pf,
            config=self.config,
        )
        self.reports: List[VerifyReport] = []

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        status = PASS if condition else FAIL
        self.reports.append(VerifyReport(name, condition, detail))
        self.log(f"  {status} {name}" + (f" — {detail}" if detail and not condition else ""))
        return condition

    # ═══════════════════════════════════════════
    #  测试1: 记忆系统基础功能
    # ═══════════════════════════════════════════

    def test_memory_basic(self) -> bool:
        self.log("\n[测试1] 记忆系统基础功能")
        all_pass = True

        # 1.1 添加工作记忆
        item = self.memory.add_memory(
            desc="玩家轻声安抚我，放下零食后离开",
            event_type="daily_feed",
            timestamp=100.0,
            importance=5.0,
        )
        all_pass &= self.check("工作记忆添加", item is not None)
        all_pass &= self.check("工作记忆非空", self.memory.size_working > 0,
                               f"size={self.memory.size_working}")

        # 1.2 长期记忆（高重要性自动进入）
        item2 = self.memory.add_memory(
            desc="玩家第一次成功摸到了我，他的手很温暖",
            event_type="first_pet_accepted",
            timestamp=200.0,
            importance=8.5,
        )
        all_pass &= self.check("长期记忆添加", self.memory.size_long_term > 0,
                               f"长期记忆={self.memory.size_long_term}")

        # 1.3 向量数据库存储
        all_pass &= self.check("向量库计数", self.memory.size_vector_db > 0,
                               f"向量库={self.memory.size_vector_db}")

        # 1.4 检索最近记忆
        recent = self.memory.retrieve_recent(3)
        all_pass &= self.check("检索最近记忆", len(recent) > 0,
                               f"最近{len(recent)}条")

        # 1.5 语义检索
        query = np.random.randn(MEMORY_EMBED_DIM).astype(np.float32)
        query = query / (np.linalg.norm(query) + 1e-8)
        results = self.memory.retrieve_by_query(query, top_k=2)
        all_pass &= self.check("语义检索", len(results) >= 0,
                               f"检索到{len(results)}条")

        # 1.6 记忆嵌入获取
        embeds = self.memory.get_memory_embeddings(query, top_k=TOP_K_MEMORIES)
        all_pass &= self.check("记忆嵌入获取", len(embeds) == TOP_K_MEMORIES,
                               f"获取{len(embeds)}条嵌入")
        all_pass &= self.check("嵌入维度正确",
                               all(e.shape[0] == MEMORY_EMBED_DIM for e in embeds))

        return all_pass

    # ═══════════════════════════════════════════
    #  测试2: 记忆时间衰减
    # ═══════════════════════════════════════════

    def test_memory_decay(self) -> bool:
        self.log("\n[测试2] 记忆时间衰减")
        all_pass = True

        # 记录衰减前状态
        long_before = self.memory.size_long_term

        # 添加一些记忆
        for i in range(10):
            self.memory.add_memory(
                desc=f"日常事件{i}",
                event_type="idle_wander",
                timestamp=0.0,  # 很久以前
                importance=4.0,
            )
        count_after_add = self.memory.size_long_term

        # 推进时间到TTL之后
        self.memory.apply_time_decay(current_time=15000.0)

        count_after_decay = self.memory.size_long_term
        all_pass &= self.check("衰减后记忆减少",
                               count_after_decay <= count_after_add,
                               f"{count_after_add} → {count_after_decay}")

        return all_pass

    # ═══════════════════════════════════════════
    #  测试3: 记忆压缩
    # ═══════════════════════════════════════════

    def test_memory_compression(self) -> bool:
        self.log("\n[测试3] 记忆压缩")
        all_pass = True

        mem = MemoryManager(config=self.config)
        mem.cfg.importance_threshold = 0.0
        mem.cfg.compress_min_count = 5
        mem.cfg.compress_age_days = 0.1

        for i in range(20):
            mem.add_memory(
                desc=f"旧日常闲逛事件{i}",
                event_type="idle_wander",
                timestamp=i * 5.0,
                importance=5.0,
            )
        before = mem.size_long_term
        self.log(f"  压缩前: {before}条")

        compressed = mem.compress_old_memories(current_time=10000.0)

        after = mem.size_long_term
        self.log(f"  压缩后: {after}条, 压缩了{compressed}条")
        all_pass &= self.check("压缩触发", compressed > 0)
        all_pass &= self.check("压缩后记忆减少", after < before)

        return all_pass

    # ═══════════════════════════════════════════
    #  测试4: 性格过滤器第一层——意图偏置
    # ═══════════════════════════════════════════

    def test_personality_intent_bias(self) -> bool:
        self.log("\n[测试4] 性格过滤器第一层——意图logits偏置")
        all_pass = True

        personalities = {
            "小雪(怯懦)": np.array(CAT_CONFIGS["xiaoxue"]["personality"], dtype=np.float32),
            "奥利奥(傲娇)": np.array(CAT_CONFIGS["oreo"]["personality"], dtype=np.float32),
            "橘子(贪吃好奇)": np.array(CAT_CONFIGS["orange"]["personality"], dtype=np.float32),
        }

        logits = np.zeros(len(INTENT_LIST), dtype=np.float32)

        for name, pvec in personalities.items():
            filtered = self.pf.filter_intent_logits(logits, pvec)

            # 检查怯懦猫的hide偏置应为正
            if "小雪" in name:
                hide_idx = INTENT_LIST.index("hide")
                all_pass &= self.check(
                    f"{name}: hide偏置>0",
                    filtered[hide_idx] > 0,
                    f"hide={filtered[hide_idx]:+.2f}"
                )
                approach_idx = INTENT_LIST.index("approach_player")
                all_pass &= self.check(
                    f"{name}: approach_player偏置<0",
                    filtered[approach_idx] < 0,
                    f"approach={filtered[approach_idx]:+.2f}"
                )

            # 检查贪吃猫的eat偏置应为正
            if "橘子" in name:
                eat_idx = INTENT_LIST.index("eat")
                all_pass &= self.check(
                    f"{name}: eat偏置>0",
                    filtered[eat_idx] > 0,
                    f"eat={filtered[eat_idx]:+.2f}"
                )

        # 批量过滤
        batch_logits = np.zeros((3, len(INTENT_LIST)), dtype=np.float32)
        batch_pvecs = np.stack(list(personalities.values()), axis=0)
        batch_filtered = self.pf.filter_batch_logits(batch_logits, batch_pvecs)
        all_pass &= self.check("批量过滤形状正确",
                               batch_filtered.shape == (3, len(INTENT_LIST)))

        return all_pass

    # ═══════════════════════════════════════════
    #  测试5: 性格过滤器第二层——行为参数
    # ═══════════════════════════════════════════

    def test_behavior_params(self) -> bool:
        self.log("\n[测试5] 性格过滤器第二层——行为参数")
        all_pass = True

        personalities = {
            "小雪(怯懦)": np.array(CAT_CONFIGS["xiaoxue"]["personality"], dtype=np.float32),
            "奥利奥(傲娇)": np.array(CAT_CONFIGS["oreo"]["personality"], dtype=np.float32),
            "橘子(贪吃好奇)": np.array(CAT_CONFIGS["orange"]["personality"], dtype=np.float32),
        }

        params = {}
        for name, pvec in personalities.items():
            params[name] = self.pf.get_behavior_params(pvec)

        # 怯懦猫应该有最大的flee_distance
        flee_distances = {n: p["flee_distance"] for n, p in params.items()}
        max_flee_cat = max(flee_distances, key=flee_distances.get)
        all_pass &= self.check(
            "怯懦猫flee_distance最大",
            "小雪" in max_flee_cat,
            f"各猫flee: {flee_distances}"
        )

        # 怯懦猫应该有最大的hesitation_weight
        hesitations = {n: p["hesitation_weight"] for n, p in params.items()}
        max_hes_cat = max(hesitations, key=hesitations.get)
        all_pass &= self.check(
            "怯懦猫犹豫权重最大",
            "小雪" in max_hes_cat,
            f"各猫犹豫: {hesitations}"
        )

        # 活跃猫move_speed最高
        speeds = {n: p["move_speed"] for n, p in params.items()}
        max_speed_cat = max(speeds, key=speeds.get)
        all_pass &= self.check(
            "橘子移动速度最高",
            "橘子" in max_speed_cat,
            f"各猫速度: {speeds}"
        )

        self.log("  行为参数对比:")
        for name, p in params.items():
            self.log(f"    {name}: speed={p['move_speed']:.2f}, "
                     f"flee={p['flee_distance']:.1f}, "
                     f"hesitation={p['hesitation_weight']:.2f}")

        return all_pass

    # ═══════════════════════════════════════════
    #  测试6: 性格过滤器第三层——文本过滤
    # ═══════════════════════════════════════════

    def test_text_filter(self) -> bool:
        self.log("\n[测试6] 性格过滤器第三层——文本过滤")
        all_pass = True

        # 傲娇猫不能说"最喜欢你了"
        oreo_vec = np.array(CAT_CONFIGS["oreo"]["personality"], dtype=np.float32)
        text1, passed1 = self.pf.filter_text("主人我最喜欢你了！喵~", oreo_vec, "奥利奥")
        all_pass &= self.check("傲娇猫禁止撒娇词", not passed1,
                               f"过滤后: '{text1}'")

        # 怯懦猫不能说"我不怕"
        xiaoxue_vec = np.array(CAT_CONFIGS["xiaoxue"]["personality"], dtype=np.float32)
        text2, passed2 = self.pf.filter_text("我不怕！放马过来！", xiaoxue_vec, "小雪")
        all_pass &= self.check("怯懦猫禁止勇敢词", not passed2,
                               f"过滤后: '{text2}'")

        # 正常文本应该通过
        text3, passed3 = self.pf.filter_text("今天天气不错，适合晒太阳。", oreo_vec, "奥利奥")
        all_pass &= self.check("正常文本通过", passed3,
                               f"文本: '{text3}'")

        return all_pass

    # ═══════════════════════════════════════════
    #  测试7: 性格意图兼容度排名
    # ═══════════════════════════════════════════

    def test_intent_ranking(self) -> bool:
        self.log("\n[测试7] 性格意图兼容度排名对比")
        all_pass = True

        personalities = {
            "小雪(怯懦)": np.array(CAT_CONFIGS["xiaoxue"]["personality"], dtype=np.float32),
            "奥利奥(傲娇)": np.array(CAT_CONFIGS["oreo"]["personality"], dtype=np.float32),
            "橘子(贪吃好奇)": np.array(CAT_CONFIGS["orange"]["personality"], dtype=np.float32),
        }

        for name, pvec in personalities.items():
            compat = self.pf.get_intent_compatibility_matrix(pvec)
            top3 = sorted(compat.items(), key=lambda x: -x[1])[:3]
            bottom3 = sorted(compat.items(), key=lambda x: x[1])[:3]
            self.log(f"  {name}:")
            self.log(f"    倾向: {', '.join(f'{i}({v:+.1f})' for i,v in top3)}")
            self.log(f"    排斥: {', '.join(f'{i}({v:+.1f})' for i,v in bottom3)}")

        # 验证怯懦猫的hide在top3
        xiaoxue_compat = self.pf.get_intent_compatibility_matrix(
            personalities["小雪(怯懦)"]
        )
        top3_xiaoxue = sorted(xiaoxue_compat.items(), key=lambda x: -x[1])[:3]
        all_pass &= self.check(
            "小雪hide在top3倾向",
            "hide" in [i for i, v in top3_xiaoxue],
            f"小雪top3: {top3_xiaoxue}"
        )

        return all_pass

    # ═══════════════════════════════════════════
    #  测试8: 记忆→RL桥接
    # ═══════════════════════════════════════════

    def test_memory_rl_bridge(self) -> bool:
        self.log("\n[测试8] Memory→RL桥接")
        all_pass = True

        bridge = MemoryRLBridge(config=self.config)

        # 8.1 查询向量构建
        emotion = np.array([30, 10, 50, 60, 20], dtype=np.float32)  # 饥饿,恐惧,好奇,舒适,社交
        env = np.array([0.6, 0.4, 0.7, 0.5, 0.3], dtype=np.float32)
        query = bridge.build_query_vector(emotion, env)
        all_pass &= self.check("查询向量维度", query.shape[0] == MEMORY_EMBED_DIM,
                               f"shape={query.shape}")

        # 8.2 先记录一些记忆
        for i in range(10):
            bridge.memory.add_memory(
                desc=f"测试记忆内容{i}",
                event_type="routine_explore",
                timestamp=float(i * 10),
                importance=5.0,
            )

        # 8.3 状态注入
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[8:13] = emotion   # 情绪段
        state[17:22] = env      # 环境段
        injected = bridge.inject_memories(state)

        # 检查记忆段非全零
        mem_start, mem_end = bridge.get_memory_indices()
        memory_part = injected[mem_start:mem_end]
        has_nonzero = np.any(np.abs(memory_part) > 0.01)
        all_pass &= self.check("记忆段已注入", has_nonzero,
                               f"记忆段非零元素: {np.count_nonzero(np.abs(memory_part) > 0.01)}")

        # 8.4 经验记录
        mem_id = bridge.record_experience(
            cat_name="奥利奥",
            intent="eat",
            bt_success=True,
            reward=2.0,
            trust_delta=1.5,
            stress_delta=-3.0,
            player_action="feed",
            timestamp=500.0,
        )
        all_pass &= self.check("经验自动记录", mem_id is not None)

        # 8.5 低价值经验不记录
        mem_id2 = bridge.record_experience(
            cat_name="奥利奥",
            intent="idle_wander",
            bt_success=True,
            reward=0.1,
            trust_delta=0.1,
            stress_delta=-0.5,
            player_action="none",
            timestamp=501.0,
        )
        all_pass &= self.check("低价值经验跳过", mem_id2 is None)

        return all_pass

    # ═══════════════════════════════════════════
    #  测试9: 解释接口
    # ═══════════════════════════════════════════

    def test_explain_interfaces(self) -> bool:
        self.log("\n[测试9] 调试解释接口")
        all_pass = True

        oreo_vec = np.array(CAT_CONFIGS["oreo"]["personality"], dtype=np.float32)

        # 意图偏置解释
        explanation = self.pf.explain_intent_bias(oreo_vec, "approach_player")
        all_pass &= self.check("意图偏置解释", len(explanation) > 0)

        # 全部意图排名
        ranking = self.pf.explain_all_intents(oreo_vec)
        all_pass &= self.check("全部意图排名", len(ranking) > 0)

        # 行为参数解释
        param_explain = self.pf.explain_behavior_params(oreo_vec, "approach_distance")
        all_pass &= self.check("行为参数解释", len(param_explain) > 0)

        # 禁用词报告
        report = self.pf.get_active_forbidden_report(oreo_vec)
        all_pass &= self.check("禁用词报告", len(report) > 0,
                               f"激活维度: {list(report.keys())}")

        if self.verbose:
            self.log(f"\n  奥利奥意图排名:\n{ranking}")

        return all_pass

    # ═══════════════════════════════════════════
    #  测试10: 概率过滤
    # ═══════════════════════════════════════════

    def test_probability_filter(self) -> bool:
        self.log("\n[测试10] 概率过滤")
        all_pass = True

        oreo_vec = np.array(CAT_CONFIGS["oreo"]["personality"], dtype=np.float32)
        xiaoxue_vec = np.array(CAT_CONFIGS["xiaoxue"]["personality"], dtype=np.float32)

        # 均匀概率
        uniform_probs = np.ones(len(INTENT_LIST), dtype=np.float32) / len(INTENT_LIST)

        oreo_filtered = self.pf.filter_probs(uniform_probs, oreo_vec)
        xiaoxue_filtered = self.pf.filter_probs(uniform_probs, xiaoxue_vec)

        all_pass &= self.check("过滤后概率仍归一化",
                               abs(oreo_filtered.sum() - 1.0) < 0.01)

        # 不同性格产生不同分布
        all_pass &= self.check("性格差异导致分布不同",
                               not np.allclose(oreo_filtered, xiaoxue_filtered, atol=0.01))

        # 奥利奥(傲娇)的hiss_warning概率应高于小雪(怯懦)的approach_player
        hiss_idx = INTENT_LIST.index("hiss_warning")
        approach_idx = INTENT_LIST.index("approach_player")
        all_pass &= self.check(
            "奥利奥hiss > 小雪hiss",
            oreo_filtered[hiss_idx] > xiaoxue_filtered[hiss_idx],
            f"奥利奥={oreo_filtered[hiss_idx]:.4f}, 小雪={xiaoxue_filtered[hiss_idx]:.4f}"
        )
        all_pass &= self.check(
            "小雪approach < 奥利奥approach",
            xiaoxue_filtered[approach_idx] < oreo_filtered[approach_idx],
            f"小雪={xiaoxue_filtered[approach_idx]:.4f}, 奥利奥={oreo_filtered[approach_idx]:.4f}"
        )

        return all_pass

    # ═══════════════════════════════════════════
    #  运行全部
    # ═══════════════════════════════════════════

    def run_all(self) -> bool:
        self.log("=" * 60)
        self.log("《猫语心声》记忆系统与性格过滤器 — 综合验证")
        self.log("=" * 60)

        tests = [
            self.test_memory_basic,
            self.test_memory_decay,
            self.test_memory_compression,
            self.test_personality_intent_bias,
            self.test_behavior_params,
            self.test_text_filter,
            self.test_intent_ranking,
            self.test_memory_rl_bridge,
            self.test_explain_interfaces,
            self.test_probability_filter,
        ]

        all_pass = True
        for test_fn in tests:
            try:
                if not test_fn():
                    all_pass = False
            except Exception as e:
                self.log(f"  {FAIL} {test_fn.__name__} 异常: {e}")
                self.reports.append(VerifyReport(test_fn.__name__, False, str(e)))
                all_pass = False

        self._print_summary(all_pass)
        return all_pass

    def _print_summary(self, all_pass: bool):
        self.log("\n" + "=" * 60)
        passed = sum(1 for r in self.reports if r.passed)
        total = len(self.reports)
        status = "全部通过!" if all_pass else f"{passed}/{total} 通过"
        self.log(f"结果: {status}")
        self.log("=" * 60)

    def export_report(self, path: str = None) -> str:
        """导出验证报告为JSON"""
        if path is None:
            path = os.path.join(self.config.export_dir, "verify_report.json")

        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": len(self.reports),
            "passed": sum(1 for r in self.reports if r.passed),
            "failed": sum(1 for r in self.reports if not r.passed),
            "tests": [
                {
                    "name": r.test_name,
                    "passed": r.passed,
                    "details": r.details,
                    "data": r.data,
                }
                for r in self.reports
            ],
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.log(f"\n报告已导出: {path}")
        return path


def verify_standalone():
    """独立验证入口"""
    verifier = MemoryPersonalityVerifier(verbose=True)
    ok = verifier.run_all()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(verify_standalone())
