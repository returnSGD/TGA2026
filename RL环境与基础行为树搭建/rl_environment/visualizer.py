"""
《猫语心声》 —— 行为树可视化调试工具

支持：
- 文本树形图（ASCII art）
- 带执行状态的彩色输出
- 行为树遍历日志
"""

from __future__ import annotations
from typing import Dict, List, Optional
from .bt_core import BTNode, BTStatus, BehaviorTree, Selector, Sequence, Parallel
from .bt_intents import get_all_behavior_trees


def print_bt_tree(bt: BehaviorTree):
    """打印行为树的文本可视化"""
    print(f"\n{'═' * 60}")
    print(f"  Behavior Tree: {bt.name}")
    print(f"  Tick count: {bt.tick_count}")
    print(f"  Last status: {_status_name(bt.last_status)}")
    print(f"{'═' * 60}")
    print(bt.tree_string())
    print()


def print_all_trees():
    """打印所有15个意图的行为树"""
    all_bts = get_all_behavior_trees()
    for intent, bt in all_bts.items():
        print_bt_tree(bt)


def print_bt_summary():
    """打印所有行为树的简要摘要"""
    all_bts = get_all_behavior_trees()
    print(f"\n{'═' * 60}")
    print(f"  宏观意图行为树一览（共{len(all_bts)}个）")
    print(f"{'═' * 60}")

    for intent, bt in all_bts.items():
        node_count = _count_nodes(bt.root)
        depth = _max_depth(bt.root)
        actions = _count_by_type(bt.root, "ActionNode|ProgressAction|NavigateAction")
        conditions = _count_by_type(bt.root, "ConditionNode")
        print(f"  {intent:22s}  Nodes:{node_count:3d}  Depth:{depth:2d}  "
              f"Actions:{actions:2d}  Conditions:{conditions:2d}")

    print(f"{'═' * 60}\n")


def visualize_bt_execution(bt: BehaviorTree, max_ticks: int = 10):
    """
    逐步执行行为树并可视化每步的状态。
    使用 \r 实现动态刷新（适合终端演示）。
    """
    import time

    bt.reset()
    print(f"\n▶ 执行行为树: {bt.name}")

    for tick in range(max_ticks):
        status = bt.tick()
        status_str = _status_name(status)

        # 清除上一行并打印当前状态
        bar = "▓" * (tick + 1) + "░" * (max_ticks - tick - 1)
        print(f"\r  Tick {tick + 1}/{max_ticks} [{bar}] {status_str}", end="")

        if status != BTStatus.RUNNING:
            print(f"\n  结果: {status_str} (tick {tick + 1})")
            break

        time.sleep(0.1)  # 减慢速度便于观察

    if bt.tick_count >= max_ticks and bt.last_status == BTStatus.RUNNING:
        print(f"\n  ⚠ 达到最大tick数 ({max_ticks})，行为树仍在运行中")


def get_bt_stats(bt: BehaviorTree) -> Dict:
    """获取行为树统计信息"""
    return {
        "name": bt.name,
        "node_count": _count_nodes(bt.root),
        "max_depth": _max_depth(bt.root),
        "selector_count": _count_by_type(bt.root, "Selector"),
        "sequence_count": _count_by_type(bt.root, "Sequence"),
        "parallel_count": _count_by_type(bt.root, "Parallel"),
        "condition_count": _count_by_type(bt.root, "ConditionNode"),
        "action_count": _count_by_type(bt.root, "ActionNode|ProgressAction|NavigateAction"),
        "decorator_count": _count_by_type(bt.root,
            "Inverter|Repeater|Timeout|Cooldown|ForceSuccess|ForceFailure|RetryUntilSuccess"),
    }


def export_bt_to_mermaid(bt: BehaviorTree) -> str:
    """将行为树导出为Mermaid流程图（可用于文档）"""
    lines = ["```mermaid", "graph TD"]
    counter = [0]

    def _add_node(node: BTNode, parent_id: str = None) -> str:
        node_id = f"N{counter[0]}"
        counter[0] += 1

        # 节点样式
        if isinstance(node, Selector):
            shape = "[/Selector/]"
        elif isinstance(node, Sequence):
            shape = "[/Sequence/]"
        elif isinstance(node, Parallel):
            shape = "[/Parallel/]"
        elif "Condition" in type(node).__name__:
            shape = "{Condition}"
        else:
            shape = f"[{type(node).__name__}]"

        lines.append(f"    {node_id}{shape}")

        if parent_id:
            lines.append(f"    {parent_id} --> {node_id}")

        for child in node.children:
            _add_node(child, node_id)

        return node_id

    _add_node(bt.root)
    lines.append("```")
    return "\n".join(lines)


# ==================== 辅助函数 ====================

def _status_name(status: BTStatus) -> str:
    return {BTStatus.SUCCESS: "✓ SUCCESS",
            BTStatus.FAILURE: "✗ FAILURE",
            BTStatus.RUNNING: "⟳ RUNNING"}.get(status, "?")


def _count_nodes(node: BTNode) -> int:
    count = 1
    for child in node.children:
        count += _count_nodes(child)
    return count


def _max_depth(node: BTNode) -> int:
    if not node.children:
        return 1
    return 1 + max(_max_depth(c) for c in node.children)


def _count_by_type(node: BTNode, type_pattern: str) -> int:
    import re
    count = 0
    node_type = type(node).__name__
    if re.search(type_pattern, node_type):
        count += 1
    for child in node.children:
        count += _count_by_type(child, type_pattern)
    return count
