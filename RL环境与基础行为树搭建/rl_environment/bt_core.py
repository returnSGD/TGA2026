"""
《猫语心声》 —— 轻量级行为树引擎

实现：Node基类, Selector(回退), Sequence(顺序), Parallel(并行),
     Decorator装饰器, Condition条件, Action动作节点
完全自研，零外部依赖，支持文本可视化调试。
"""

from __future__ import annotations
from enum import IntEnum
from typing import Any, Dict, List, Optional, Callable
import time
import random


class BTStatus(IntEnum):
    SUCCESS = 0
    FAILURE = 1
    RUNNING = 2


class Blackboard:
    """行为树共享数据黑板"""

    def __init__(self):
        self._data: Dict[str, Any] = {}

    def set(self, key: str, value: Any):
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._data

    def delete(self, key: str):
        self._data.pop(key, None)

    def clear(self):
        self._data.clear()

    def dump(self) -> Dict[str, Any]:
        return dict(self._data)


class BTNode:
    """行为树节点基类"""

    def __init__(self, name: str = "", children: Optional[List[BTNode]] = None):
        self.name = name or self.__class__.__name__
        self.parent: Optional[BTNode] = None
        self.children: List[BTNode] = []
        if children:
            for child in children:
                self.add_child(child)

    def add_child(self, child: BTNode) -> BTNode:
        child.parent = self
        self.children.append(child)
        return self

    def tick(self, blackboard: Blackboard) -> BTStatus:
        raise NotImplementedError

    def reset(self):
        """重置节点内部状态"""
        for child in self.children:
            child.reset()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def tree_string(self, indent: int = 0, is_last: bool = True, prefix: str = "") -> str:
        """生成文本形式的树结构（用于可视化调试）"""
        lines = []
        connector = "└── " if is_last else "├── "
        status_icon = self._status_icon()
        lines.append(f"{prefix}{connector}{status_icon}{self.name}")

        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(self.children):
            child_is_last = (i == len(self.children) - 1)
            lines.append(child.tree_string(indent + 1, child_is_last, child_prefix))
        return "\n".join(lines)

    def _status_icon(self) -> str:
        return ""


class Selector(BTNode):
    """
    选择器（回退节点）：依次执行子节点，遇到SUCCESS或RUNNING则停止。
    所有子节点FAILURE则返回FAILURE。
    """

    def __init__(self, name: str = "Selector", children=None):
        super().__init__(name, children=children)
        self._current_idx = 0

    def tick(self, blackboard: Blackboard) -> BTStatus:
        for i in range(self._current_idx, len(self.children)):
            child = self.children[i]
            status = child.tick(blackboard)
            if status == BTStatus.RUNNING:
                self._current_idx = i
                return BTStatus.RUNNING
            if status == BTStatus.SUCCESS:
                self._current_idx = 0
                return BTStatus.SUCCESS
        self._current_idx = 0
        return BTStatus.FAILURE

    def reset(self):
        self._current_idx = 0
        super().reset()

    def _status_icon(self) -> str:
        return "?"


class Sequence(BTNode):
    """
    顺序节点：依次执行子节点，遇到FAILURE或RUNNING则停止。
    所有子节点SUCCESS则返回SUCCESS。
    """

    def __init__(self, name: str = "Sequence", children=None):
        super().__init__(name, children=children)
        self._current_idx = 0

    def tick(self, blackboard: Blackboard) -> BTStatus:
        for i in range(self._current_idx, len(self.children)):
            child = self.children[i]
            status = child.tick(blackboard)
            if status == BTStatus.RUNNING:
                self._current_idx = i
                return BTStatus.RUNNING
            if status == BTStatus.FAILURE:
                self._current_idx = 0
                return BTStatus.FAILURE
        self._current_idx = 0
        return BTStatus.SUCCESS

    def reset(self):
        self._current_idx = 0
        super().reset()

    def _status_icon(self) -> str:
        return "→"


class Parallel(BTNode):
    """
    并行节点：同时执行所有子节点。
    policy: "all_success" | "any_success" | "all_fail"
    """

    def __init__(self, name: str = "Parallel",
                 policy: str = "all_success",
                 max_children: int = 0, children=None):
        super().__init__(name, children=children)
        self.policy = policy
        self._running_mask: List[bool] = []

    def tick(self, blackboard: Blackboard) -> BTStatus:
        if not self._running_mask:
            self._running_mask = [True] * len(self.children)

        success_count = 0
        failure_count = 0
        for i, child in enumerate(self.children):
            if not self._running_mask[i]:
                continue
            status = child.tick(blackboard)
            if status == BTStatus.SUCCESS:
                self._running_mask[i] = False
                success_count += 1
            elif status == BTStatus.FAILURE:
                self._running_mask[i] = False
                failure_count += 1

        if self.policy == "any_success" and success_count > 0:
            self._running_mask = []
            return BTStatus.SUCCESS
        if self.policy == "all_success" and success_count == len(self.children):
            self._running_mask = []
            return BTStatus.SUCCESS
        if failure_count == len(self.children):
            self._running_mask = []
            return BTStatus.FAILURE
        return BTStatus.RUNNING

    def reset(self):
        self._running_mask = []
        super().reset()

    def _status_icon(self) -> str:
        return "⇉"


# ==================== 装饰器节点 ====================

class Inverter(BTNode):
    """取反：SUCCESS→FAILURE, FAILURE→SUCCESS, RUNNING不变"""

    def __init__(self, name: str = "Inverter", children=None):
        super().__init__(name, children=children)

    def tick(self, blackboard: Blackboard) -> BTStatus:
        if not self.children:
            return BTStatus.FAILURE
        status = self.children[0].tick(blackboard)
        if status == BTStatus.SUCCESS:
            return BTStatus.FAILURE
        if status == BTStatus.FAILURE:
            return BTStatus.SUCCESS
        return BTStatus.RUNNING

    def _status_icon(self) -> str:
        return "¬"


class Repeater(BTNode):
    """重复执行子节点n次或直到FAILURE"""

    def __init__(self, name: str = "Repeater", times: int = -1,
                 until_fail: bool = False, children=None):
        super().__init__(name, children=children)
        self.times = times
        self.until_fail = until_fail
        self._count = 0

    def tick(self, blackboard: Blackboard) -> BTStatus:
        if not self.children:
            return BTStatus.FAILURE

        child = self.children[0]
        status = child.tick(blackboard)

        if self.until_fail:
            if status == BTStatus.FAILURE:
                self._count = 0
                return BTStatus.SUCCESS
            if status == BTStatus.SUCCESS:
                self._count += 1
                child.reset()
                return BTStatus.RUNNING
            return BTStatus.RUNNING
        else:
            if status == BTStatus.RUNNING:
                return BTStatus.RUNNING
            self._count += 1
            child.reset()
            if self.times > 0 and self._count >= self.times:
                self._count = 0
                return status
            return BTStatus.RUNNING

    def reset(self):
        self._count = 0
        super().reset()

    def _status_icon(self) -> str:
        return "↻"


class RetryUntilSuccess(BTNode):
    """重复执行直到成功，最多max_attempts次"""

    def __init__(self, name: str = "RetryUntilSuccess", max_attempts: int = 3,
                 children=None):
        super().__init__(name, children=children)
        self.max_attempts = max_attempts
        self._attempts = 0

    def tick(self, blackboard: Blackboard) -> BTStatus:
        if not self.children:
            return BTStatus.FAILURE
        if self._attempts >= self.max_attempts:
            self._attempts = 0
            return BTStatus.FAILURE

        status = self.children[0].tick(blackboard)
        if status == BTStatus.SUCCESS:
            self._attempts = 0
            return BTStatus.SUCCESS
        if status == BTStatus.FAILURE:
            self._attempts += 1
            self.children[0].reset()
            if self._attempts < self.max_attempts:
                return BTStatus.RUNNING
            self._attempts = 0
            return BTStatus.FAILURE
        return BTStatus.RUNNING

    def reset(self):
        self._attempts = 0
        super().reset()

    def _status_icon(self) -> str:
        return "⟳"


class ForceSuccess(BTNode):
    """无论子节点返回什么都返回SUCCESS"""

    def __init__(self, name: str = "ForceSuccess", children=None):
        super().__init__(name, children=children)

    def tick(self, blackboard: Blackboard) -> BTStatus:
        if not self.children:
            return BTStatus.SUCCESS
        status = self.children[0].tick(blackboard)
        if status == BTStatus.RUNNING:
            return BTStatus.RUNNING
        return BTStatus.SUCCESS

    def _status_icon(self) -> str:
        return "✓"


class ForceFailure(BTNode):
    """无论子节点返回什么都返回FAILURE"""

    def __init__(self, name: str = "ForceFailure", children=None):
        super().__init__(name, children=children)

    def tick(self, blackboard: Blackboard) -> BTStatus:
        if not self.children:
            return BTStatus.FAILURE
        status = self.children[0].tick(blackboard)
        if status == BTStatus.RUNNING:
            return BTStatus.RUNNING
        return BTStatus.FAILURE

    def _status_icon(self) -> str:
        return "✗"


class Timeout(BTNode):
    """限时执行：超时则强制FAILURE"""

    def __init__(self, name: str = "Timeout", seconds: float = 5.0, children=None):
        super().__init__(name, children=children)
        self.seconds = seconds
        self._start_time: Optional[float] = None

    def tick(self, blackboard: Blackboard) -> BTStatus:
        if not self.children:
            return BTStatus.FAILURE
        if self._start_time is None:
            self._start_time = time.time()
        if time.time() - self._start_time > self.seconds:
            self._start_time = None
            return BTStatus.FAILURE
        status = self.children[0].tick(blackboard)
        if status != BTStatus.RUNNING:
            self._start_time = None
        return status

    def reset(self):
        self._start_time = None
        super().reset()

    def _status_icon(self) -> str:
        return "⏱"


class Cooldown(BTNode):
    """冷却装饰器：成功执行后进入冷却，冷却期间直接返回FAILURE"""

    def __init__(self, name: str = "Cooldown", cooldown_ticks: int = 5,
                 children=None):
        super().__init__(name, children=children)
        self.cooldown_ticks = cooldown_ticks
        self._ticks_since_last = cooldown_ticks

    def tick(self, blackboard: Blackboard) -> BTStatus:
        if not self.children:
            return BTStatus.FAILURE
        if self._ticks_since_last < self.cooldown_ticks:
            self._ticks_since_last += 1
            return BTStatus.FAILURE
        status = self.children[0].tick(blackboard)
        if status == BTStatus.SUCCESS:
            self._ticks_since_last = 0
        return status

    def reset(self):
        self._ticks_since_last = self.cooldown_ticks
        super().reset()

    def _status_icon(self) -> str:
        return "❄"


class RandomChoice(BTNode):
    """随机选择一个子节点执行（加权随机）"""

    def __init__(self, name: str = "RandomChoice",
                 weights: Optional[List[float]] = None, children=None):
        super().__init__(name, children=children)
        self.weights = weights
        self._selected: Optional[int] = None

    def tick(self, blackboard: Blackboard) -> BTStatus:
        if not self.children:
            return BTStatus.FAILURE
        if self._selected is None:
            w = self.weights if self.weights else [1.0] * len(self.children)
            self._selected = random.choices(range(len(self.children)), weights=w, k=1)[0]
        status = self.children[self._selected].tick(blackboard)
        if status != BTStatus.RUNNING:
            self._selected = None
        return status

    def reset(self):
        self._selected = None
        super().reset()

    def _status_icon(self) -> str:
        return "🎲"


# ==================== 条件节点 ====================

class ConditionNode(BTNode):
    """条件节点：执行check函数，返回SUCCESS/FAILURE（从不RUNNING）"""

    def __init__(self, name: str, check_fn: Callable[[Blackboard], bool],
                 children=None):
        super().__init__(name, children=children)
        self.check_fn = check_fn

    def tick(self, blackboard: Blackboard) -> BTStatus:
        return BTStatus.SUCCESS if self.check_fn(blackboard) else BTStatus.FAILURE

    def _status_icon(self) -> str:
        return "◆"


# ==================== 动作节点 ====================

class ActionNode(BTNode):
    """动作节点：执行action_fn，可能返回RUNNING表示进行中"""

    def __init__(self, name: str,
                 action_fn: Callable[[Blackboard], BTStatus],
                 on_enter: Optional[Callable[[Blackboard], None]] = None,
                 on_exit: Optional[Callable[[Blackboard], None]] = None,
                 children=None):
        super().__init__(name, children=children)
        self.action_fn = action_fn
        self.on_enter = on_enter
        self.on_exit = on_exit
        self._entered = False

    def tick(self, blackboard: Blackboard) -> BTStatus:
        if not self._entered and self.on_enter:
            self.on_enter(blackboard)
            self._entered = True
        status = self.action_fn(blackboard)
        if status != BTStatus.RUNNING:
            if self._entered and self.on_exit:
                self.on_exit(blackboard)
            self._entered = False
        return status

    def reset(self):
        self._entered = False
        super().reset()

    def _status_icon(self) -> str:
        return "▶"


# ==================== 行为树运行器 ====================

class BehaviorTree:
    """行为树运行器，管理整棵树的tick、调试与可视化"""

    def __init__(self, root: BTNode, name: str = "BT"):
        self.root = root
        self.name = name
        self.blackboard = Blackboard()
        self.last_status = BTStatus.FAILURE
        self.tick_count = 0
        self._log: List[str] = []

    def tick(self) -> BTStatus:
        self.tick_count += 1
        self.last_status = self.root.tick(self.blackboard)
        return self.last_status

    def reset(self):
        self.root.reset()
        self.blackboard.clear()
        self.last_status = BTStatus.FAILURE
        self.tick_count = 0
        self._log.clear()

    def log(self, msg: str):
        self._log.append(f"[tick {self.tick_count}] {msg}")

    def get_log(self) -> List[str]:
        return list(self._log)

    def tree_string(self) -> str:
        """生成行为树文本可视化"""
        lines = [f"BehaviorTree: {self.name}"]
        lines.append("═" * 40)
        lines.append(self.root.tree_string())
        return "\n".join(lines)

    def print_tree(self):
        print(self.tree_string())


def build_bt_from_desc(name: str, nodes_desc: list, parent: BTNode = None) -> BTNode:
    """
    从描述列表快捷构建行为树。
    nodes_desc 格式: [("type", "name", {params}, [children...]), ...]

    示例:
        build_bt_from_desc("eat_bt", [
            ("Sequence", "main", {}, [
                ("ConditionNode", "hungry?", {"check_fn": lambda bb: bb.get("hunger") > 50}, []),
                ("ActionNode", "go_eat", {"action_fn": lambda bb: ...}, []),
            ]),
        ])
    """
    # 在实际使用中，bt_intents.py 会直接使用节点类构建，此函数为辅助工具
    pass
