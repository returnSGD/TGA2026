"""
《猫语心声》 —— 记忆系统与性格过滤器 CLI

用法:
    python -m memory_personality.main              # 运行全部验证
    python -m memory_personality.main --demo       # 演示记忆系统
    python -m memory_personality.main --contrast   # 性格对比演示
    python -m memory_personality.main --profile    # 性格画像展示
    python -m memory_personality.main --export     # 导出验证报告
"""

from __future__ import annotations
import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_personality.config import MemoryConfig, CAT_PERSONALITIES
from memory_personality.memory_manager import MemoryManager
from memory_personality.personality_filter import PersonalityFilter
from memory_personality.memory_rl_bridge import MemoryRLBridge
from memory_personality.verify import MemoryPersonalityVerifier
from rl_environment.config import (
    INTENT_LIST, PERSONALITY_KEYS, CAT_CONFIGS,
    INTENT_PERSONALITY_MATRIX, PERSONALITY_BEHAVIOR_PARAMS,
)


def cmd_verify(args):
    """运行全部验证测试"""
    verifier = MemoryPersonalityVerifier(verbose=not args.quiet)
    ok = verifier.run_all()
    if args.export:
        verifier.export_report(args.export)
    return 0 if ok else 1


def cmd_demo(args):
    """演示记忆系统完整工作流"""
    print("=" * 60)
    print("《猫语心声》记忆系统演示")
    print("=" * 60)

    config = MemoryConfig()
    memory = MemoryManager(config=config)

    # 模拟游戏事件序列
    events = [
        (0, "清晨，咖啡厅开门，阳光洒进大厅", "routine_explore", 3.0),
        (10, "玩家走进咖啡厅，轻声呼唤猫咪们", "routine_explore", 3.5),
        (20, "玩家在食盆里放入新鲜猫粮", "daily_feed", 5.0),
        (30, "奥利奥远远观察玩家的一举一动", "routine_explore", 3.0),
        (40, "玩家蹲下来，向奥利奥伸出手", "routine_pet_accepted", 6.0),
        (50, "奥利奥犹豫了一下，闻了闻玩家的手指", "first_pet_accepted", 8.0),
        (60, "玩家轻轻抚摸奥利奥的下巴", "routine_pet_accepted", 6.5),
        (70, "奥利奥发出低沉的呼噜声", "milestone_first_purr", 9.5),
        (80, "小雪从躲藏处探出头，好奇地张望", "routine_explore", 4.0),
        (90, "玩家发现小雪，保持距离安静坐下", "player_arranged_comfort", 6.5),
        (100, "橘子狼吞虎咽吃完猫粮，开始四处探索", "routine_explore", 3.0),
        (110, "橘子发现玩具老鼠，兴奋地扑上去", "played_with_player", 5.5),
        (120, "奥利奥主动蹭了蹭玩家的腿", "first_voluntary_rub", 9.0),
        (130, "傍晚，玩家打扫咖啡厅后离开", "routine_explore", 3.0),
        (140, "猫咪们各自找位置准备入睡", "routine_sleep", 2.5),
    ]

    print("\n[事件序列录入]")
    for tick, desc, event_type, importance in events:
        memory.add_memory(
            desc=desc,
            event_type=event_type,
            timestamp=float(tick),
            importance=importance,
        )
        print(f"  tick {tick:3d} | {event_type:30s} | imp={importance:.1f} | {desc}")

    print(f"\n[记忆统计] {memory.summary()}")

    # 检索演示
    print("\n[语义检索演示] 查询: '玩家抚摸猫咪'")
    query_embed = memory.embed_service.encode("玩家温柔地抚摸猫咪")
    results = memory.retrieve_by_query(query_embed, top_k=3)
    for i, mem in enumerate(results):
        print(f"  {i+1}. [{mem.event_type}] imp={mem.importance:.1f} | {mem.desc}")

    # 时间衰减
    print("\n[时间衰减] 推进30天后...")
    memory.apply_time_decay(current_time=30 * 144.0)
    print(f"  {memory.summary()}")

    # 事件分布
    print("\n[事件类型分布]")
    for event_type, count in memory.get_event_distribution().items():
        bar = "#" * min(count, 40)
        print(f"  {event_type:30s} {count:3d} {bar}")

    # 导出
    if args.export:
        export_path = args.export
        state = memory.export_to_dict()
        os.makedirs(os.path.dirname(export_path) if os.path.dirname(export_path) else ".", exist_ok=True)
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"\n记忆状态已导出: {export_path}")

    return 0


def cmd_contrast(args):
    """性格差异对比演示"""
    print("=" * 60)
    print("《猫语心声》三猫性格对比")
    print("=" * 60)

    pf = PersonalityFilter()

    cats = {
        "小雪 (怯懦型)": np.array(CAT_CONFIGS["xiaoxue"]["personality"], dtype=np.float32),
        "奥利奥 (傲娇型)": np.array(CAT_CONFIGS["oreo"]["personality"], dtype=np.float32),
        "橘子 (贪吃好奇型)": np.array(CAT_CONFIGS["orange"]["personality"], dtype=np.float32),
    }

    # 性格画像
    print("\n[性格维度雷达]")
    header = f"{'维度':8s}"
    for name in cats:
        header += f" {name[:6]:>10s}"
    print(header)
    print("-" * (8 + 11 * len(cats)))
    for j, trait in enumerate(PERSONALITY_KEYS):
        row = f"{trait:8s}"
        for vec in cats.values():
            row += f" {vec[j]:10.2f}"
        print(row)

    # 意图倾向对比
    print("\n[意图倾向对比] (Top-5 + Bottom-3)")
    for name, vec in cats.items():
        compat = pf.get_intent_compatibility_matrix(vec)
        sorted_intents = sorted(compat.items(), key=lambda x: -x[1])
        top5 = sorted_intents[:5]
        bottom3 = sorted_intents[-3:]

        print(f"\n  {name}:")
        print(f"    ▲ 倾向: " + ", ".join(
            f"{i}({v:+.1f})" for i, v in top5
        ))
        print(f"    ▼ 排斥: " + ", ".join(
            f"{i}({v:+.1f})" for i, v in bottom3
        ))

    # 行为参数对比
    print("\n[行为参数对比]")
    param_names = ["approach_distance", "move_speed", "response_delay",
                   "flee_distance", "hesitation_weight"]
    header = f"{'参数':22s}"
    for name in cats:
        header += f" {name[:6]:>10s}"
    print(header)
    print("-" * (22 + 11 * len(cats)))
    for pname in param_names:
        row = f"{pname:22s}"
        for name, vec in cats.items():
            params = pf.get_behavior_params(vec)
            row += f" {params[pname]:10.2f}"
        print(row)

    # 禁用词激活
    print("\n[禁用词激活报告]")
    for name, vec in cats.items():
        report = pf.get_active_forbidden_report(vec)
        if report:
            active_traits = list(report.keys())
            total_forbidden = sum(r["forbidden_count"] for r in report.values())
            print(f"  {name}: 激活维度={active_traits}, 总禁用词={total_forbidden}")
        else:
            print(f"  {name}: 无激活维度")

    return 0


def cmd_profile(args):
    """单猫性格画像"""
    cat_id = args.cat
    cfg = CAT_CONFIGS.get(cat_id)
    if cfg is None:
        print(f"未知猫咪: {cat_id}，可选: {list(CAT_CONFIGS.keys())}")
        return 1

    pf = PersonalityFilter()
    vec = np.array(cfg["personality"], dtype=np.float32)

    print("=" * 60)
    print(f"  {cfg['name']} ({cfg['breed']}) — 性格画像")
    print("=" * 60)
    print(f"\n背景: {cfg['backstory']}")
    print(f"特征: {cfg['traits']}")
    print(f"初始信任: {cfg['trust_init']}  初始压力: {cfg['stress_init']}")

    print(f"\n[性格维度]")
    for j, trait in enumerate(PERSONALITY_KEYS):
        bar = "█" * int(vec[j] * 20) + "░" * (20 - int(vec[j] * 20))
        print(f"  {trait:6s} [{bar}] {vec[j]:.2f}")

    print(f"\n{'-' * 40}")
    print(pf.explain_all_intents(vec))

    print(f"\n[行为参数]")
    params = pf.get_behavior_params(vec)
    for k, v in params.items():
        print(f"  {k}: {v:.2f}")

    print(f"\n[禁用词]")
    report = pf.get_active_forbidden_report(vec)
    if report:
        for trait, info in report.items():
            print(f"  {trait}: {info['forbidden_count']}个禁用词")
            for word in info["sample"]:
                print(f"    - {word}")
    else:
        print("  无激活维度")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="《猫语心声》记忆系统与性格过滤器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # verify
    p_verify = sub.add_parser("verify", help="运行验证测试")
    p_verify.add_argument("--quiet", "-q", action="store_true", help="减少输出")
    p_verify.add_argument("--export", "-e", type=str, help="导出报告路径")

    # demo
    p_demo = sub.add_parser("demo", help="记忆系统演示")
    p_demo.add_argument("--export", "-e", type=str, help="导出路径")

    # contrast
    sub.add_parser("contrast", help="三猫性格对比")

    # profile
    p_profile = sub.add_parser("profile", help="单猫性格画像")
    p_profile.add_argument("--cat", "-c", type=str, default="oreo",
                           choices=["xiaoxue", "oreo", "orange"],
                           help="猫咪ID")

    args = parser.parse_args()

    if args.command == "verify":
        return cmd_verify(args)
    elif args.command == "demo":
        return cmd_demo(args)
    elif args.command == "contrast":
        return cmd_contrast(args)
    elif args.command == "profile":
        return cmd_profile(args)
    else:
        # 默认：运行全部
        print("未指定命令，运行全部验证...\n")
        return cmd_verify(argparse.Namespace(quiet=False, export=None))


if __name__ == "__main__":
    sys.exit(main())
