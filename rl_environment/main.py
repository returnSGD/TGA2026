"""
《猫语心声》 —— RL环境与基础行为树搭建
主运行入口（阶段一：第1-4周）

运行沙盒模拟器，测试所有意图的行为树，收集训练数据。

用法:
    # 快速测试（200 tick）
    python -m rl_environment.main

    # 收集 10000+ 条训练数据并自动导出
    python -m rl_environment.main --target-samples 10000 --export

    # 调试模式
    python -m rl_environment.main --test-all
    python -m rl_environment.main --test-personality
    python -m rl_environment.main --debug-bt eat
"""

from __future__ import annotations
import sys
import os
import random
import argparse
import time
import numpy as np
from typing import Dict, List, Optional

# 确保项目路径在sys.path中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_environment.config import (
    INTENT_LIST, CAT_CONFIGS, PERSONALITY_KEYS,
    TICKS_PER_DAY, DAY_PHASES, ACTION_DURATION, ROOMS, MEMORY_EMBED_DIM,
)
from rl_environment.environment import SandboxEnvironment
from rl_environment.cat_state import CatState, MemoryManager
from rl_environment.cat_agent import CatAgent
from rl_environment.personality_filter import PersonalityFilter
from rl_environment.data_collector import DataCollector
from rl_environment.bt_intents import get_all_behavior_trees, build_bt_for_intent
from rl_environment.bt_core import BTStatus
from rl_environment.visualizer import (
    print_bt_tree, print_bt_summary, print_all_trees,
    visualize_bt_execution, get_bt_stats,
)

# ==================== 动态玩家行为生成 ====================

# 玩家行为权重分布（模拟真实玩家的操作频率）
PLAYER_ACTION_WEIGHTS = {
    "none":      35,   # 大部分时间不操作
    "approach":   8,   # 偶尔走动
    "pet":       12,   # 抚摸是最常见互动
    "call":       6,   # 呼唤
    "feed":       8,   # 喂食
    "treat":      5,   # 给零食
    "play":       7,   # 玩耍
    "leave":      4,   # 离开
    "photo":      2,   # 拍照（少见）
    "scold":      1,   # 斥责（极少）
}

# 根据昼夜阶段调整行为概率
PHASE_ACTION_MODIFIERS = {
    "morning":   {"feed": 1.5, "call": 1.3, "play": 0.5, "none": 0.8},
    "afternoon": {"pet": 1.5, "play": 1.3, "feed": 0.7, "none": 0.7},
    "evening":   {"treat": 1.8, "pet": 1.5, "call": 1.2, "none": 0.8},
    "night":     {"none": 2.5, "pet": 0.2, "play": 0.1, "call": 0.3, "photo": 0.0},
}


def generate_player_action(tick: int, day_phase: str, rng: random.Random) -> str:
    """
    根据当前游戏阶段动态生成玩家行为（概率加权随机）。
    确保长线模拟中玩家行为的多样性和合理性。
    """
    modifiers = PHASE_ACTION_MODIFIERS.get(day_phase, {})
    actions = list(PLAYER_ACTION_WEIGHTS.keys())
    weights = []
    for a in actions:
        w = PLAYER_ACTION_WEIGHTS[a] * modifiers.get(a, 1.0)
        weights.append(max(0.1, w))

    return rng.choices(actions, weights=weights, k=1)[0]


def create_cats(personality_filter: PersonalityFilter) -> Dict[str, CatAgent]:
    """根据配置创建三只猫咪"""
    cats = {}
    for cat_id, cfg in CAT_CONFIGS.items():
        agent = CatAgent(cat_id=cat_id, config=cfg,
                        personality_filter=personality_filter)
        cats[cat_id] = agent
    return cats


def init_cat_positions(env: SandboxEnvironment, cats: Dict[str, CatAgent]):
    """初始化猫咪在环境中的位置"""
    positions = {
        "xiaoxue": (1, 2),    # 小雪躲在角落（怯懦）
        "oreo":    (6, 4),    # 奥利奥在大厅中央（傲娇）
        "orange":  (3, 6),    # 橘子在大厅中活跃
    }
    for cat_id, agent in cats.items():
        pos = positions.get(cat_id, (5, 5))
        agent.state.position = pos
        agent.state.current_room_id = env.get_room_id_at(pos)
        env.cat_positions[cat_id] = pos


def build_state_vector(cat: CatAgent, player_action: str) -> np.ndarray:
    """为数据收集构建完整的状态向量（422维）"""
    # 获取记忆嵌入
    state_vec_full = cat.state.to_state_vector()
    query_vec = np.zeros(MEMORY_EMBED_DIM, dtype=np.float32)
    query_vec[:len(state_vec_full)] = state_vec_full
    memory_embeds = cat.memory_mgr.get_memory_embeddings(query_vec)
    return cat.build_full_state(player_action, memory_embeds)


# ==================== 主模拟循环 ====================

def run_simulation(ticks: int = 200, visualize: bool = False,
                   verbose: bool = True, export: bool = False,
                   target_samples: int = 0, seed: int = 42):
    """
    运行沙盒模拟。

    参数:
        ticks: 最大 tick 数（如果 target_samples > 0 则可能提前结束或延长）
        target_samples: 目标收集样本数（0 表示不限制）
        export: 结束后导出数据
        verbose: 详细输出模式
        seed: 随机种子
    """
    rng = random.Random(seed)

    print(f"\n{'═' * 70}")
    print(f"  《猫语心声》RL环境与行为树 — 沙盒模拟器 v2.0")
    print(f"  基于技术策划案v2 (HRLTM架构) 阶段一：数据收集")
    print(f"{'═' * 70}")

    if target_samples > 0:
        # 预估需要 tick 数：3只猫 × ticks ≈ 3×ticks 条转移
        est_ticks = max(ticks, target_samples // 3 + 50)
        print(f"  目标样本数: {target_samples}+  |  预估需 {est_ticks}+ ticks")
        ticks = max(ticks, est_ticks)

    # ---- 初始化 ----
    env = SandboxEnvironment(seed=seed)
    pf = PersonalityFilter()
    cats = create_cats(pf)
    init_cat_positions(env, cats)
    collector = DataCollector(save_dir="./training_data", mode="bc")

    print(f"\n已创建 {len(cats)} 只猫咪:")
    for cid, cat in cats.items():
        name = cat.name
        breed = cat.config.get("breed", "?")
        traits = cat.config.get("traits", "?")
        personality = cat.personality_summary
        print(f"  {name} ({cid}) - {breed}")
        print(f"     性格: [{personality}]")
        print(f"     特征: {traits}")
        print(f"     信任:{cat.state.trust_level:.0f} 压力:{cat.state.stress_level:.0f}")

    print(f"\n环境: 网格{env.grid.shape[1]}×{env.grid.shape[0]}, "
          f"{len(env.objects)}个物体, {len(ROOMS)}个房间")
    print(f"时间: 第{env.day_count}天 {env.day_phase} 天气:{env.weather}")
    print(f"玩家行为: 动态概率生成 (受昼夜阶段影响)")

    # ---- 行为树预构建 ----
    print(f"\n{'─' * 70}")
    print("  行为树构建与验证")
    print(f"{'─' * 70}")

    all_bts = get_all_behavior_trees()
    for intent, bt in all_bts.items():
        stats = get_bt_stats(bt)
        print(f"  {intent:22s} → Nodes:{stats['node_count']:3d} "
              f"Depth:{stats['max_depth']:2d}")

    # ---- 可视化（可选） ----
    if visualize:
        print(f"\n{'─' * 70}")
        print("  行为树可视化")
        print(f"{'─' * 70}")
        key_intents = ["eat", "hide", "approach_player", "fearful_retreat"]
        for intent in key_intents:
            bt = all_bts.get(intent)
            if bt:
                print_bt_tree(bt)
        print_bt_summary()

    # ---- 初始化数据收集 ----
    # 为每只猫创建独立的 episode
    for cid in cats:
        collector.start_episode(cid)

    # 存储"上一tick的状态向量"用于完成 pending transition
    # {cat_id: state_vector_before_interaction}
    prev_states: Dict[str, np.ndarray] = {}

    total_reward = 0.0
    start_time = time.time()
    last_progress_tick = 0
    auto_export_threshold = 5000  # 每收集5000条自动导出一次

    print(f"\n{'═' * 70}")
    if target_samples > 0:
        print(f"  开始收集训练数据（目标: ≥{target_samples} 条有效转移）")
    else:
        print(f"  模拟开始（共{ticks}个tick）")
    print(f"{'═' * 70}")

    if verbose and ticks <= 500:
        print(f"{'Tick':>4} {'Phase':>8} {'Player':>8} | "
              f"{'小雪(怯懦)':>30s} | {'奥利奥(傲娇)':>30s} | {'橘子(乐天)':>30s}")
        print("-" * 120)

    tick = 0
    while tick < ticks:
        env.advance_tick()
        player_action = generate_player_action(tick, env.day_phase, rng)
        env.set_player_action(player_action)

        tick_results = {}

        for cid, cat in cats.items():
            # ── Step 1: 构建当前 tick 的状态向量（在交互之前） ──
            current_state = build_state_vector(cat, player_action)

            # ── Step 2: 完成上一 tick 的 pending transition ──
            #  用当前状态作为上一 tick 的 next_state
            if cid in prev_states:
                collector.complete_pending(
                    cid,
                    next_state=current_state,
                    done=False,
                )
                # 记录上一 tick 的 state/action 为 pending
                # （已在上一轮中通过 record_pending 设置）

            # ── Step 3: 执行本次交互决策 ──
            result = cat.process_interaction(env, player_action=player_action)
            tick_results[cid] = result
            total_reward += result["reward"]

            # ── Step 4: 记录本次的 (state, action) 为 pending ──
            collector.record_pending(
                cat_id=cid,
                state=current_state,
                intent=result["intent"],
                reward=result["reward"],
                info={"cat_id": cid, "bt_status": int(result["bt_status"]),
                      "tick": tick, "day_phase": env.day_phase},
            )

            # ── Step 5: 保存当前状态，供下一 tick 作为 prev_state ──
            prev_states[cid] = current_state

        # ── 日志输出（采样输出以减少IO开销） ──
        log_interval = 50 if ticks > 500 else 5
        if verbose and tick % log_interval == 0:
            phase_short = env.day_phase[:2]
            action_short = player_action[:5]
            parts = []
            for cid in ["xiaoxue", "oreo", "orange"]:
                r = tick_results.get(cid, {})
                intent = r.get("intent", "?")[:8]
                bt_stat = {BTStatus.SUCCESS: "✓", BTStatus.FAILURE: "✗",
                          BTStatus.RUNNING: "⟳"}.get(r.get("bt_status"), "?")
                cat = cats[cid]
                parts.append(f"{intent:8s}{bt_stat} T={cat.state.trust_level:3.0f}")
            print(f"{tick:4d} {phase_short:>8s} {action_short:>8s} | "
                  f"{' | '.join(parts)}")

        # ── 进度报告（长运行模式） ──
        if ticks > 200 and tick - last_progress_tick >= 200:
            elapsed = time.time() - start_time
            rate = collector.valid_transitions / elapsed if elapsed > 0 else 0
            print(f"\n--- 进度: tick {tick}/{ticks} | {collector.progress_summary()} "
                  f"| 速率: {rate:.1f} 条/秒 | 耗时: {elapsed:.0f}s ---\n")
            last_progress_tick = tick

            # 达到目标提前结束
            if target_samples > 0 and collector.valid_transitions >= target_samples:
                print(f"\n  ✓ 已达到目标样本数 {target_samples}！提前结束模拟。")
                break

        # ── 定期自动导出（长运行模式防数据丢失） ──
        if (ticks > 500 and export and
            collector.valid_transitions > 0 and
            collector.valid_transitions % auto_export_threshold == 0 and
            collector.valid_transitions > 0):
            collector.export_bc_data(
                filename=f"bc_data_autosave_{collector.valid_transitions}samples.npz"
            )

        tick += 1

    # ── 模拟结束，flush 所有 pending transitions ──
    #  用当前状态的副本（零向量填充）完成最后一条
    for cid, cat in cats.items():
        final_state = build_state_vector(cat, "none")
        collector.complete_pending(cid, final_state, done=True)

    elapsed = time.time() - start_time

    # ---- 模拟结束报告 ----
    print(f"\n{'═' * 70}")
    print(f"  模拟完成（耗时 {elapsed:.0f}s, {tick} ticks）")
    print(f"{'═' * 70}")

    # ---- 猫咪统计 ----
    print(f"\n{'─' * 70}")
    print("  猫咪统计")
    print(f"{'─' * 70}")
    for cid, cat in cats.items():
        print(f"  {cat.stats_summary()}")
    print(f"\n  总奖励: {total_reward:.2f}")
    print(f"  数据收集速率: {collector.valid_transitions / elapsed:.1f} 条/秒")

    # ---- 训练数据统计 ----
    print(f"\n{'─' * 70}")
    print(collector.stats_report())
    print(f"{'─' * 70}")

    # ---- 数据导出 ----
    if export:
        collector.export_bc_data()
        collector.export_csv()

    # ---- 记忆系统状态 ----
    print(f"\n{'─' * 70}")
    print("  记忆系统状态")
    print(f"{'─' * 70}")
    for cid, cat in cats.items():
        print(f"  {cat.name}: {cat.memory_mgr.summary()}")

    # ---- 数据覆盖度检查 ----
    unique_intents = sum(1 for c in collector.action_counts.values() if c > 0)
    total_intents = len(INTENT_LIST)
    print(f"\n{'─' * 70}")
    print(f"  数据覆盖度: {unique_intents}/{total_intents} 种意图被收集"
          f"({unique_intents/total_intents:.0%})")
    if unique_intents < total_intents:
        missing = [INTENT_LIST[i] for i, c in collector.action_counts.items() if c == 0]
        print(f"  缺失意图: {missing}")
        print(f"  提示: 增加模拟tick数以覆盖更多场景")
    print(f"{'─' * 70}")

    print(f"\n{'═' * 70}")
    if collector.valid_transitions >= 10000:
        print(f"  ✓ 已达到 ≥10000 条有效训练数据！可以进入阶段二。")
    else:
        remaining = 10000 - collector.valid_transitions
        print(f"  当前 {collector.valid_transitions} 条，还需约 {remaining} 条。")
        print(f"  建议: python -m rl_environment.main --target-samples 10000 --export")
    print(f"{'═' * 70}\n")

    return env, cats, collector


# ==================== 行为树调试模式 ====================

def debug_bt_execution(intent: str = "eat"):
    """调试单个行为树的逐步执行过程"""
    from rl_environment.bt_core import Blackboard
    from rl_environment.bt_intents import build_bt_for_intent
    from rl_environment.visualizer import visualize_bt_execution, print_bt_tree

    env = SandboxEnvironment()
    pf = PersonalityFilter()
    cats = create_cats(pf)
    init_cat_positions(env, cats)

    cat = cats.get("oreo")
    bt = build_bt_for_intent(intent)

    bt.blackboard.set("state", cat.state)
    bt.blackboard.set("env", env)
    bt.blackboard.set("cat_id", cat.cat_id)
    bt.blackboard.set("behavior_params", cat.behavior_params)

    print_bt_tree(bt)
    visualize_bt_execution(bt, max_ticks=20)


def test_all_intents():
    """遍历所有意图，测试行为树能否正常完成或正确失败"""
    env = SandboxEnvironment()
    pf = PersonalityFilter()
    cats = create_cats(pf)
    init_cat_positions(env, cats)

    print(f"\n{'═' * 70}")
    print(f"  所有意图行为树完整性测试")
    print(f"{'═' * 70}")

    for intent in INTENT_LIST:
        bt = build_bt_for_intent(intent)
        bt.reset()
        cat = cats["oreo"]
        bt.blackboard.set("state", cat.state)
        bt.blackboard.set("env", env)
        bt.blackboard.set("cat_id", cat.cat_id)
        bt.blackboard.set("behavior_params", cat.behavior_params)

        final_status = BTStatus.RUNNING
        for tick in range(30):
            status = bt.tick()
            if status != BTStatus.RUNNING:
                final_status = status
                break

        status_str = {BTStatus.SUCCESS: "✓ OK", BTStatus.FAILURE: "✗ FAIL",
                     BTStatus.RUNNING: "⟳ TIMEOUT"}.get(final_status, "?")
        print(f"  {intent:22s} → {status_str} (tick {bt.tick_count})")


def test_personality_differences():
    """测试同一场景下不同性格猫咪的意图选择差异"""
    env = SandboxEnvironment()
    pf = PersonalityFilter()
    cats = create_cats(pf)
    init_cat_positions(env, cats)

    print(f"\n{'═' * 70}")
    print(f"  性格差异测试：同一场景，不同意图")
    print(f"{'═' * 70}")

    env.set_player_action("approach")
    env.player_position = (5, 5)

    for cid, cat in cats.items():
        cat.state.position = (4, 4)
        cat.state.hunger = 0.3
        cat.state.fear = 0.35
        cat.state.curiosity = 0.5
        cat.state.energy = 0.6

    for cid, cat in cats.items():
        intent = cat.decide_intent_with_rule(env)
        bias_info = pf.explain_intent_bias(cat.state.personality_vector, "approach_player")
        params = pf.get_behavior_params(cat.state.personality_vector)
        print(f"\n  {cat.name} ({cat.personality_summary})")
        print(f"     选择意图: {intent}")
        print(f"     approach_player偏置分析:")
        for line in bias_info.split("\n"):
            print(f"       {line}")
        print(f"     行为参数: 靠近距离={params['approach_distance']:.1f}m "
              f"移动速度={params['move_speed']:.1f} 犹豫度={params['hesitation_weight']:.1f}")


# ==================== CLI入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="《猫语心声》RL环境与基础行为树 — 沙盒模拟器"
    )
    parser.add_argument("--ticks", type=int, default=200,
                       help="最大模拟tick数（默认200）")
    parser.add_argument("--target-samples", type=int, default=0,
                       help="目标收集样本数（如 10000），达到后自动停止")
    parser.add_argument("--visualize", action="store_true",
                       help="输出行为树可视化")
    parser.add_argument("--export", action="store_true",
                       help="模拟结束后导出训练数据")
    parser.add_argument("--quiet", action="store_true",
                       help="安静模式，减少输出")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子（默认42）")
    parser.add_argument("--debug-bt", type=str, metavar="INTENT",
                       help="调试指定意图的行为树（如：eat, hide）")
    parser.add_argument("--test-all", action="store_true",
                       help="测试所有意图的行为树")
    parser.add_argument("--test-personality", action="store_true",
                       help="测试性格差异")

    args = parser.parse_args()

    if args.debug_bt:
        debug_bt_execution(args.debug_bt)
    elif args.test_all:
        test_all_intents()
    elif args.test_personality:
        test_personality_differences()
    else:
        run_simulation(
            ticks=args.ticks,
            visualize=args.visualize,
            verbose=not args.quiet,
            export=args.export,
            target_samples=args.target_samples,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
