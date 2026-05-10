"""
《猫语心声》 —— RL环境与基础行为树搭建
主运行入口（阶段一：第1-4周）

运行沙盒模拟器，测试所有意图的行为树，收集训练数据。

用法:
    python -m rl_environment.main
    python -m rl_environment.main --ticks 500 --visualize
    python -m rl_environment.main --ticks 1000 --export
"""

from __future__ import annotations
import sys
import os
import random
import argparse
import numpy as np
from typing import Dict, List

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


# ==================== 模拟场景预设 ====================

PLAYER_ACTIONS_SCHEDULE = [
    # (start_tick, duration, action, description)
    (0,   10,  "none",      "观察猫咖"),
    (10,  8,   "approach",  "玩家走进大厅"),
    (20,  5,   "call",      "玩家呼唤猫咪"),
    (30,  10,  "feed",      "玩家添加猫粮"),
    (42,  6,   "pet",       "玩家抚摸猫咪"),
    (50,  15,  "none",      "玩家观察"),
    (70,  5,   "play",      "玩家拿出逗猫棒"),
    (80,  10,  "none",      "让猫咪自由活动"),
    (95,  5,   "treat",     "玩家发放零食"),
    (105, 8,   "none",      "观察反应"),
    (120, 5,   "call",      "再次呼唤"),
    (130, 14,  "none",      "自由活动时间"),
]


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


def get_player_action(tick: int) -> str:
    """根据预设时间表获取玩家动作"""
    for start, duration, action, _ in PLAYER_ACTIONS_SCHEDULE:
        if start <= tick < start + duration:
            return action
    return "none"


# ==================== 主模拟循环 ====================

def run_simulation(ticks: int = 200, visualize: bool = False,
                   verbose: bool = True, export: bool = False):
    """运行沙盒模拟"""

    print(f"\n{'═' * 70}")
    print(f"  《猫语心声》RL环境与行为树 — 沙盒模拟器 v1.0")
    print(f"  基于技术策划案v2 (HRLTM架构) 阶段一实现")
    print(f"{'═' * 70}")

    # ---- 初始化 ----
    env = SandboxEnvironment(seed=42)
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
        print(f"  🐱 {name} ({cid}) - {breed}")
        print(f"     性格: [{personality}]")
        print(f"     特征: {traits}")
        print(f"     信任:{cat.state.trust_level:.0f} 压力:{cat.state.stress_level:.0f}")

    print(f"\n环境: 网格{env.grid.shape[1]}×{env.grid.shape[0]}, "
          f"{len(env.objects)}个物体, {len(ROOMS)}个房间")
    print(f"时间: 第{env.day_count}天 {env.day_phase} 天气:{env.weather}")

    # ---- 行为树预构建和验证 ----
    print(f"\n{'─' * 70}")
    print("  行为树构建与验证")
    print(f"{'─' * 70}")

    all_bts = get_all_behavior_trees()
    for intent, bt in all_bts.items():
        stats = get_bt_stats(bt)
        print(f"  {intent:22s} → Nodes:{stats['node_count']:3d} "
              f"Depth:{stats['max_depth']:2d} "
              f"Sel:{stats['selector_count']} Seq:{stats['sequence_count']} "
              f"Cond:{stats['condition_count']} Act:{stats['action_count']}")

    # ---- 可视化（可选） ----
    if visualize:
        print(f"\n{'─' * 70}")
        print("  行为树可视化")
        print(f"{'─' * 70}")

        # 展示几个关键意图的行为树
        key_intents = ["eat", "hide", "approach_player", "fearful_retreat"]
        for intent in key_intents:
            bt = all_bts.get(intent)
            if bt:
                print_bt_tree(bt)

        print_bt_summary()

    # ---- 主模拟循环 ----
    print(f"\n{'═' * 70}")
    print(f"  模拟开始（共{ticks}个tick）")
    print(f"{'═' * 70}")

    if verbose:
        print(f"{'Tick':>4} {'Phase':>8} {'Player':>8} | "
              f"{'小雪(怯懦)':>30s} | {'奥利奥(傲娇)':>30s} | {'橘子(乐天)':>30s}")
        print("-" * 120)

    # 为每只猫初始化episode
    for cid in cats:
        collector.start_episode(cid)

    total_reward = 0.0

    for tick in range(ticks):
        env.advance_tick()
        player_action = get_player_action(tick)
        env.set_player_action(player_action)

        tick_results = {}

        # 每只猫决策
        for cid, cat in cats.items():
            result = cat.process_interaction(env, player_action=player_action)
            tick_results[cid] = result
            total_reward += result["reward"]

            # 收集训练数据
            if tick > 0 and tick < ticks - 1:
                state_vec_full = cat.state.to_state_vector()
                query_vec = np.zeros(MEMORY_EMBED_DIM, dtype=np.float32)
                query_vec[:len(state_vec_full)] = state_vec_full
                memory_embeds = cat.memory_mgr.get_memory_embeddings(query_vec)
                state_vec = cat.build_full_state(player_action, memory_embeds)
                next_state_vec = state_vec.copy()
                collector.record_step(
                    state=state_vec,
                    intent=result["intent"],
                    reward=result["reward"],
                    next_state=next_state_vec,
                    done=(tick >= ticks - 1),
                    info={"cat_id": cid, "bt_status": int(result["bt_status"])},
                )

        # 日志输出
        if verbose and (tick % 5 == 0 or tick < 10 or player_action != "none"
                        or any(r["bt_status"] != BTStatus.RUNNING for r in tick_results.values())):
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

        # 每50 tick输出环境状态
        if verbose and tick % 50 == 0 and tick > 0:
            print(f"\n--- Tick {tick} 环境状态 ---")
            print(f"  时间: Day {env.day_count} {env.day_phase} 天气: {env.weather}")
            for cid, cat in cats.items():
                print(f"  {cat.name}: {cat.state.summary()}")
            print(f"  累计奖励: {total_reward:.2f}")
            print()

    # ---- 模拟结束 ----
    print(f"\n{'═' * 70}")
    print(f"  模拟完成")
    print(f"{'═' * 70}")

    # ---- 统计报告 ----
    print(f"\n{'─' * 70}")
    print("  猫咪统计")
    print(f"{'─' * 70}")
    for cid, cat in cats.items():
        print(f"  {cat.stats_summary()}")
    print(f"\n  总奖励: {total_reward:.2f}")

    # ---- 训练数据统计 ----
    print(f"\n{'─' * 70}")
    print(collector.stats_report())
    print(f"{'─' * 70}")

    # ---- 数据导出 ----
    if export:
        collector.export_bc_data()
        collector.export_csv()

    # ---- 内存使用印象 ----
    print(f"\n{'─' * 70}")
    print("  记忆系统状态")
    print(f"{'─' * 70}")
    for cid, cat in cats.items():
        print(f"  {cat.name}: {cat.memory_mgr.summary()}")

    print(f"\n{'═' * 70}")
    print(f"  沙盒模拟器运行完毕。")
    print(f"  下一步：将收集的训练数据用于阶段二RL策略网络训练。")
    print(f"{'═' * 70}\n")

    return env, cats, collector


# ==================== 行为树调试模式 ====================

def debug_bt_execution(intent: str = "eat"):
    """调试单个行为树的逐步执行过程"""
    from rl_environment.bt_core import Blackboard
    from rl_environment.bt_intents import build_bt_for_intent
    from rl_environment.visualizer import visualize_bt_execution, print_bt_tree

    # 创建迷你环境
    env = SandboxEnvironment()
    pf = PersonalityFilter()
    cats = create_cats(pf)
    init_cat_positions(env, cats)

    cat = cats.get("oreo")
    bt = build_bt_for_intent(intent)

    # 准备黑板
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

        # 使用奥利奥的状态
        cat = cats["oreo"]
        bt.blackboard.set("state", cat.state)
        bt.blackboard.set("env", env)
        bt.blackboard.set("cat_id", cat.cat_id)
        bt.blackboard.set("behavior_params", cat.behavior_params)

        # 执行最多30 tick
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

    # 场景：玩家靠近
    env.set_player_action("approach")
    env.player_position = (5, 5)

    # 统一设置相似状态
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
        print(f"\n  🐱 {cat.name} ({cat.personality_summary})")
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
                       help="模拟tick数（默认200）")
    parser.add_argument("--visualize", action="store_true",
                       help="输出行为树可视化")
    parser.add_argument("--export", action="store_true",
                       help="导出训练数据")
    parser.add_argument("--quiet", action="store_true",
                       help="安静模式，减少输出")
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
        )


if __name__ == "__main__":
    main()
