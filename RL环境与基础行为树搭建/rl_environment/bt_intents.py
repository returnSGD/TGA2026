"""
《猫语心声》 —— 15个宏观意图的行为树子图

每个意图对应一棵完整的行为树，包含：
- 前提条件检查
- 主执行序列
- 失败/中断处理
- 参数化（通过黑板接收性格参数）

行为树最终返回 SUCCESS/FAILURE，驱动动画状态机和数值更新。
"""

from __future__ import annotations
from typing import Callable, Dict, Optional
import random

from .bt_core import (
    BTNode, BTStatus, Blackboard,
    Selector, Sequence, Parallel,
    ConditionNode, ActionNode,
    Inverter, Repeater, RetryUntilSuccess,
    ForceSuccess, ForceFailure, Timeout, Cooldown,
    BehaviorTree,
)
from .config import INTENT_LIST, ACTION_DURATION


# ==================== 辅助函数 ====================

def _check_hunger(bb: Blackboard) -> bool:
    state = bb.get("state")
    threshold = bb.get("hunger_threshold", 0.5)
    return state is not None and state.hunger > threshold


def _check_energy(bb: Blackboard) -> bool:
    state = bb.get("state")
    threshold = bb.get("energy_threshold", 0.3)
    return state is not None and state.energy < threshold


def _check_fear(bb: Blackboard) -> bool:
    state = bb.get("state")
    threshold = bb.get("fear_threshold", 0.6)
    return state is not None and state.fear > threshold


def _check_curiosity(bb: Blackboard) -> bool:
    state = bb.get("state")
    threshold = bb.get("curiosity_threshold", 0.6)
    return state is not None and state.curiosity > threshold


def _check_trust(bb: Blackboard) -> bool:
    state = bb.get("state")
    threshold = bb.get("trust_threshold", 30)
    return state is not None and state.trust_level >= threshold


def _check_player_nearby(bb: Blackboard) -> bool:
    env = bb.get("env")
    state = bb.get("state")
    if env is None or state is None:
        return False
    if not hasattr(env, 'manhattan_distance'):
        return False
    dist = env.manhattan_distance(state.position, env.player_position)
    return dist <= bb.get("player_dist_threshold", 3)


def _check_food_available(bb: Blackboard) -> bool:
    env = bb.get("env")
    state = bb.get("state")
    if env is None or state is None:
        return False
    obj = env.get_nearest_object(state.position, "food_bowl")
    return obj is not None


def _check_toy_available(bb: Blackboard) -> bool:
    env = bb.get("env")
    state = bb.get("state")
    if env is None or state is None:
        return False
    toys = [o for o in env.objects.values() if o.is_toy]
    return len(toys) > 0


def _check_hiding_available(bb: Blackboard) -> bool:
    env = bb.get("env")
    state = bb.get("state")
    if env is None or state is None:
        return False
    obj = env.get_nearest_object(state.position, "hiding_box")
    return obj is not None


def _check_bed_available(bb: Blackboard) -> bool:
    env = bb.get("env")
    state = bb.get("state")
    if env is None or state is None:
        return False
    obj = env.get_nearest_object(state.position, "cat_bed")
    return obj is not None


def _check_nearby_cats(bb: Blackboard) -> bool:
    env = bb.get("env")
    cat_id = bb.get("cat_id", "")
    if env is None or not cat_id:
        return False
    nearby = env.get_nearby_cats(cat_id, max_dist=3)
    bb.set("_nearby_cats", nearby)
    return len(nearby) > 0


def _check_window_available(bb: Blackboard) -> bool:
    env = bb.get("env")
    state = bb.get("state")
    if env is None or state is None:
        return False
    obj = env.get_nearest_object(state.position, "window_spot")
    return obj is not None


def _check_is_safe(bb: Blackboard) -> bool:
    state = bb.get("state")
    return state is not None and state.fear < 0.5 and state.stress_level < 60


def _check_player_action(bb: Blackboard) -> bool:
    env = bb.get("env")
    target_action = bb.get("target_player_action", "pet")
    return env is not None and env.player_action == target_action


def _check_consecutive_failures(bb: Blackboard) -> bool:
    state = bb.get("state")
    return state is not None and state.consecutive_intent_failures >= 3


# ==================== 通用动作节点工厂 ====================

class NavigateAction(ActionNode):
    """导航到目标（通用移动节点）"""

    def __init__(self, name: str, target_key: str = "navigate_target",
                 speed_key: str = "move_speed"):
        self.target_key = target_key
        self.speed_key = speed_key
        super().__init__(name, action_fn=self._navigate_fn,
                        on_enter=self._on_navigate_enter)

    def _on_navigate_enter(self, bb: Blackboard):
        bb.set("_nav_step", 0)
        bb.set("_nav_path", None)

    def _navigate_fn(self, bb: Blackboard) -> BTStatus:
        env = bb.get("env")
        state = bb.get("state")
        target = bb.get(self.target_key)
        if env is None or state is None or target is None:
            return BTStatus.FAILURE

        if env.manhattan_distance(state.position, target) <= 1:
            state.position = target
            return BTStatus.SUCCESS

        step = bb.get("_nav_step", 0)
        new_pos = env.move_toward(state.position, target, steps=1)
        if new_pos == state.position:
            return BTStatus.FAILURE  # 无法移动

        state.position = new_pos
        bb.set("_nav_step", step + 1)

        # 更新猫在环境中的位置
        cat_id = bb.get("cat_id", "")
        if cat_id:
            env.cat_positions[cat_id] = new_pos
            state.current_room_id = env.get_room_id_at(new_pos)

        # 检查是否到达
        if env.manhattan_distance(new_pos, target) <= 1:
            state.position = target
            return BTStatus.SUCCESS
        return BTStatus.RUNNING


class ProgressAction(ActionNode):
    """带进度的持续动作（如进食、睡眠）"""

    def __init__(self, name: str, total_ticks: int,
                 state_update_fn: Optional[Callable] = None,
                 on_complete: Optional[Callable] = None):
        self.total_ticks = total_ticks
        self.state_update_fn = state_update_fn
        self.on_complete_fn = on_complete
        super().__init__(name, action_fn=self._progress_fn,
                        on_enter=self._on_progress_enter,
                        on_exit=self._on_progress_exit)

    def _on_progress_enter(self, bb: Blackboard):
        bb.set("_progress", 0)

    def _progress_fn(self, bb: Blackboard) -> BTStatus:
        progress = bb.get("_progress", 0)
        progress += 1

        if self.state_update_fn:
            self.state_update_fn(bb)

        if progress >= self.total_ticks:
            if self.on_complete_fn:
                self.on_complete_fn(bb)
            return BTStatus.SUCCESS

        bb.set("_progress", progress)
        return BTStatus.RUNNING

    def _on_progress_exit(self, bb: Blackboard):
        pass


# ==================== 15个意图的行为树构建 ====================

def build_idle_wander_bt() -> BehaviorTree:
    """
    闲逛/发呆
    结构: Selector[
        Condition[连续失败过多] → ForceSuccess[Action[发呆]],
        Sequence[
            Action[随机选择方向移动1-3格],
            Action[短暂停留(发呆2-3 tick)],
        ]
    ]
    """
    root = Selector("idle_wander_root")

    # 连续失败过多时强制发呆（降级行为）
    fail_branch = Sequence("force_idle")
    fail_branch.add_child(ConditionNode("too_many_failures?", _check_consecutive_failures))
    fail_branch.add_child(ForceSuccess("force_idle_ok", [
        ProgressAction("idle_forced", ACTION_DURATION["move"] * 3)
    ]))
    root.add_child(fail_branch)

    # 正常闲逛
    wander_seq = Sequence("wander_main")
    wander_seq.add_child(ActionNode("pick_random_direction", action_fn=lambda bb: (
        bb.set("navigate_target", _random_nearby_pos(bb)),
        BTStatus.SUCCESS
    )))
    wander_seq.add_child(NavigateAction("move_random", "navigate_target"))
    wander_seq.add_child(ProgressAction("pause_idle", random.randint(2, 4)))
    root.add_child(wander_seq)

    bt = BehaviorTree(root, "idle_wander")
    return bt


def build_approach_player_bt() -> BehaviorTree:
    """
    主动靠近玩家
    结构: Sequence[
        Condition[玩家在场 & 信任足够],
        Action[寻路至玩家2米内],
        Action[减速靠近(按性格参数调整距离)],
        Action[等待回应, 超时3 tick],
    ]
    """
    root = Sequence("approach_player_main")

    root.add_child(ConditionNode("player_nearby_check", lambda bb: (
        bb.get("env") is not None and _check_player_nearby(bb)
    )))
    root.add_child(ConditionNode("trust_enough", lambda bb: (
        _check_trust(bb)
    )))

    # 寻路到玩家附近
    root.add_child(ActionNode("set_target_player", action_fn=lambda bb: (
        bb.set("navigate_target", bb.get("env").player_position),
        BTStatus.SUCCESS
    )))
    root.add_child(NavigateAction("move_to_player", "navigate_target"))

    # 减速靠近（根据性格行为参数调整距离）
    root.add_child(ActionNode("slow_approach", action_fn=_slow_approach_fn))

    # 等待回应
    root.add_child(Timeout("wait_response", seconds=3.0))
    root.add_child(ProgressAction("stand_near", 3))

    bt = BehaviorTree(root, "approach_player")
    return bt


def _slow_approach_fn(bb: Blackboard) -> BTStatus:
    """减速靠近玩家"""
    env = bb.get("env")
    state = bb.get("state")
    params = bb.get("behavior_params", {})
    approach_dist = params.get("approach_distance", 1.5)

    if env is None or state is None:
        return BTStatus.FAILURE

    player_pos = env.player_position
    dist = env.manhattan_distance(state.position, player_pos)

    # 到达合适距离
    if dist <= max(1, int(approach_dist)):
        return BTStatus.SUCCESS

    # 继续靠近
    new_pos = env.move_toward(state.position, player_pos, steps=1)
    if new_pos == state.position:
        return BTStatus.SUCCESS
    state.position = new_pos
    cat_id = bb.get("cat_id", "")
    if cat_id:
        env.cat_positions[cat_id] = new_pos
        state.current_room_id = env.get_room_id_at(new_pos)
    return BTStatus.RUNNING


def build_ask_for_attention_bt() -> BehaviorTree:
    """撒娇/蹭玩家/求关注"""
    root = Sequence("ask_attention_main")
    root.add_child(ConditionNode("trust_high", lambda bb: (
        _check_trust(bb) and bb.get("state").trust_level >= 50
    )))
    root.add_child(ConditionNode("player_nearby", _check_player_nearby))
    root.add_child(ActionNode("move_close", action_fn=lambda bb: (
        bb.set("navigate_target", bb.get("env").player_position),
        BTStatus.SUCCESS
    )))
    root.add_child(NavigateAction("approach_player", "navigate_target"))
    root.add_child(ActionNode("rub_against", action_fn=_rub_against_fn))
    root.add_child(ProgressAction("wait_reaction", 3))

    bt = BehaviorTree(root, "ask_for_attention")
    return bt


def _rub_against_fn(bb: Blackboard) -> BTStatus:
    """蹭玩家动作"""
    state = bb.get("state")
    if state is not None:
        state.comfort += 0.05
        state.social_need -= 0.1
    return BTStatus.SUCCESS


def build_eat_bt() -> BehaviorTree:
    """
    进食
    结构: Selector[
        Sequence[
            Condition[饥饿>50 & 食物可用],
            Action[寻路至食盆],
            Action[进食(持续5 tick)],
        ],
        Action[report_no_food: FAILURE],
    ]
    """
    root = Selector("eat_root")

    main_seq = Sequence("eat_main")
    main_seq.add_child(ConditionNode("is_hungry", lambda bb: (
        bb.get("state") is not None and bb.get("state").hunger > 0.5
    )))
    main_seq.add_child(ConditionNode("food_exists", _check_food_available))
    main_seq.add_child(ActionNode("find_food", action_fn=lambda bb: (
        _find_nearest_and_set_target(bb, "food_bowl"),
        BTStatus.SUCCESS
    )))
    main_seq.add_child(NavigateAction("move_to_food", "navigate_target"))
    main_seq.add_child(ActionNode("start_eating", action_fn=lambda bb: (
        _set_flag(bb, "set_eating", True),
        BTStatus.SUCCESS
    ), on_exit=lambda bb: _set_flag(bb, "set_eating", False)))
    main_seq.add_child(ProgressAction("eat_process", ACTION_DURATION["eat"],
        state_update_fn=_eating_update))
    main_seq.add_child(ActionNode("finish_eating", action_fn=lambda bb: (
        _set_flag(bb, "set_eating", False),
        BTStatus.SUCCESS
    )))
    root.add_child(main_seq)

    # 无食物
    root.add_child(Sequence("no_food", [
        ActionNode("report_no_food", action_fn=lambda bb: (
            _set_flag(bb, "report_msg", "找不到食物"),
            BTStatus.FAILURE
        ))
    ]))

    bt = BehaviorTree(root, "eat")
    return bt


def _eating_update(bb: Blackboard):
    """进食过程中的状态更新"""
    state = bb.get("state")
    if state:
        state.hunger -= 15.0 / 100.0
        state.comfort += 2.0 / 100.0
        state.stress_level -= 2.0


def build_sleep_bt() -> BehaviorTree:
    """睡觉/打盹"""
    root = Selector("sleep_root")

    main_seq = Sequence("sleep_main")
    main_seq.add_child(ConditionNode("low_energy", lambda bb: (
        bb.get("state") is not None and bb.get("state").energy < 0.3
    )))
    main_seq.add_child(ConditionNode("safe_enough", _check_is_safe))

    # 优先找床，找不到就地睡
    bed_or_spot = Selector("find_bed_or_spot")
    bed_seq = Sequence("find_bed")
    bed_seq.add_child(ConditionNode("bed_available", _check_bed_available))
    bed_seq.add_child(ActionNode("find_bed", action_fn=lambda bb: (
        _find_nearest_and_set_target(bb, "cat_bed"),
        BTStatus.SUCCESS
    )))
    bed_seq.add_child(NavigateAction("move_to_bed", "navigate_target"))
    bed_or_spot.add_child(bed_seq)
    bed_or_spot.add_child(ForceSuccess("sleep_here", [
        ProgressAction("rest_where_i_am", 2)
    ]))
    main_seq.add_child(bed_or_spot)

    main_seq.add_child(ActionNode("start_sleeping", action_fn=lambda bb: (
        _set_flag(bb, "set_sleeping", True),
        BTStatus.SUCCESS
    ), on_exit=lambda bb: _set_flag(bb, "set_sleeping", False)))
    main_seq.add_child(ProgressAction("sleep_process", ACTION_DURATION["sleep"],
        state_update_fn=_sleeping_update))
    main_seq.add_child(ActionNode("wake_up", action_fn=lambda bb: (
        _set_flag(bb, "set_sleeping", False),
        BTStatus.SUCCESS
    )))
    root.add_child(main_seq)

    # 不能睡（不安全等）
    root.add_child(Sequence("cant_sleep", [
        ActionNode("stay_alert", action_fn=lambda bb: (
            _set_flag(bb, "report_msg", "太不安全了，不能睡"),
            BTStatus.FAILURE
        ))
    ]))

    bt = BehaviorTree(root, "sleep")
    return bt


def _sleeping_update(bb: Blackboard):
    state = bb.get("state")
    if state:
        state.energy += 8.0 / 100.0
        state.comfort += 3.0 / 100.0
        state.stress_level -= 1.5


def build_play_with_toy_bt() -> BehaviorTree:
    """玩指定玩具"""
    root = Selector("play_root")

    main_seq = Sequence("play_main")
    main_seq.add_child(ConditionNode("toy_available", _check_toy_available))
    main_seq.add_child(ActionNode("find_toy", action_fn=lambda bb: (
        _find_nearest_and_set_target(bb, "toy_mouse", "toy_ball"),
        BTStatus.SUCCESS
    )))
    main_seq.add_child(NavigateAction("move_to_toy", "navigate_target"))
    main_seq.add_child(ProgressAction("play_process", ACTION_DURATION["play"],
        state_update_fn=_playing_update))
    main_seq.add_child(ForceSuccess("play_done", [
        ProgressAction("play_afterglow", random.randint(1, 3))
    ]))
    root.add_child(main_seq)

    root.add_child(Sequence("no_toy", [
        ActionNode("report_no_toy", action_fn=lambda bb: (
            _set_flag(bb, "report_msg", "没有可玩的玩具"),
            BTStatus.FAILURE
        ))
    ]))

    bt = BehaviorTree(root, "play_with_toy")
    return bt


def _playing_update(bb: Blackboard):
    state = bb.get("state")
    if state:
        state.energy -= 5.0 / 100.0
        state.comfort += 4.0 / 100.0
        state.curiosity -= 0.1
        state.stress_level -= 5.0


def build_social_groom_bt() -> BehaviorTree:
    """与某只猫互相舔毛"""
    root = Selector("social_groom_root")

    main_seq = Sequence("groom_main")
    main_seq.add_child(ConditionNode("other_cats_nearby", _check_nearby_cats))
    main_seq.add_child(ActionNode("pick_partner", action_fn=_pick_social_partner))
    main_seq.add_child(ActionNode("set_target_cat", action_fn=lambda bb: (
        _set_target_to_cat(bb, bb.get("_social_target", "")),
        BTStatus.SUCCESS
    )))
    main_seq.add_child(NavigateAction("move_to_partner", "navigate_target"))
    main_seq.add_child(ProgressAction("groom_process", ACTION_DURATION["groom"],
        state_update_fn=_grooming_update))
    root.add_child(main_seq)

    root.add_child(Sequence("no_partner", [
        ActionNode("report_alone", action_fn=lambda bb: (
            _set_flag(bb, "report_msg", "没有可以互相舔毛的伙伴"),
            BTStatus.FAILURE
        ))
    ]))

    bt = BehaviorTree(root, "social_groom")
    return bt


def _grooming_update(bb: Blackboard):
    state = bb.get("state")
    if state:
        state.social_need -= 0.15
        state.comfort += 0.05
        state.stress_level -= 3.0


def build_social_play_bt() -> BehaviorTree:
    """与某只猫玩耍/追逐"""
    root = Selector("social_play_root")

    main_seq = Sequence("play_chase_main")
    main_seq.add_child(ConditionNode("other_cats_nearby", _check_nearby_cats))
    main_seq.add_child(ConditionNode("has_energy", lambda bb: (
        bb.get("state") is not None and bb.get("state").energy > 0.5
    )))
    main_seq.add_child(ActionNode("pick_playmate", action_fn=_pick_social_partner))
    main_seq.add_child(ProgressAction("chase_play", ACTION_DURATION["play"],
        state_update_fn=_social_play_update))
    root.add_child(main_seq)

    root.add_child(Sequence("no_playmate", [
        ActionNode("report_no_playmate", action_fn=lambda bb: (
            _set_flag(bb, "report_msg", "没有精力或玩伴"),
            BTStatus.FAILURE
        ))
    ]))

    bt = BehaviorTree(root, "social_play")
    return bt


def _social_play_update(bb: Blackboard):
    state = bb.get("state")
    if state:
        state.social_need -= 0.12
        state.energy -= 6.0 / 100.0
        state.comfort += 3.0 / 100.0
        state.curiosity += 0.03
        state.stress_level -= 4.0


def build_hide_bt() -> BehaviorTree:
    """躲藏"""
    root = Selector("hide_root")

    main_seq = Sequence("hide_main")
    main_seq.add_child(ConditionNode("is_scared", lambda bb: (
        bb.get("state") is not None and (
            bb.get("state").fear > 0.5 or bb.get("state").stress_level > 70
        )
    )))
    main_seq.add_child(ActionNode("find_hiding_spot", action_fn=lambda bb: (
        _find_nearest_and_set_target(bb, "hiding_box"),
        BTStatus.SUCCESS
    )))
    main_seq.add_child(NavigateAction("move_to_hiding", "navigate_target"))
    main_seq.add_child(ActionNode("enter_hiding", action_fn=lambda bb: (
        _set_flag(bb, "set_hiding", True),
        BTStatus.SUCCESS
    ), on_exit=lambda bb: _set_flag(bb, "set_hiding", False)))
    main_seq.add_child(ProgressAction("hide_process", ACTION_DURATION["hide"] * 3,
        state_update_fn=_hiding_update))
    root.add_child(main_seq)

    # 无躲藏点但需要躲：就地蜷缩
    root.add_child(Sequence("huddle_in_place", [
        ConditionNode("must_hide", lambda bb: (
            bb.get("state") is not None and bb.get("state").fear > 0.7
        )),
        ActionNode("huddle", action_fn=lambda bb: (
            _set_flag(bb, "set_hiding", True),
            BTStatus.SUCCESS
        )),
        ProgressAction("huddle_in_corner", ACTION_DURATION["hide"] * 5,
            state_update_fn=_hiding_update),
        ActionNode("relax", action_fn=lambda bb: (
            _set_flag(bb, "set_hiding", False),
            BTStatus.SUCCESS
        )),
    ]))

    bt = BehaviorTree(root, "hide")
    return bt


def _hiding_update(bb: Blackboard):
    state = bb.get("state")
    if state:
        state.fear -= 5.0 / 100.0
        state.stress_level -= 2.0
        state.comfort -= 1.0 / 100.0


def build_hiss_warning_bt() -> BehaviorTree:
    """发出警告/哈气"""
    root = Sequence("hiss_main")
    root.add_child(ConditionNode("feels_threatened", lambda bb: (
        bb.get("state") is not None and (
            bb.get("state").fear > 0.6 or
            (hasattr(bb.get("env"), 'player_action') and
             bb.get("env").player_action in ("approach", "grab"))
        )
    )))
    root.add_child(ActionNode("hiss", action_fn=_hiss_fn))
    root.add_child(ProgressAction("stand_ground", ACTION_DURATION["hiss"]))

    bt = BehaviorTree(root, "hiss_warning")
    return bt


def _hiss_fn(bb: Blackboard) -> BTStatus:
    state = bb.get("state")
    if state:
        state.fear -= 0.05
        state.stress_level += 3.0
    return BTStatus.SUCCESS


def build_curious_inspect_bt() -> BehaviorTree:
    """好奇探索某物/某处"""
    root = Selector("inspect_root")

    main_seq = Sequence("inspect_main")
    main_seq.add_child(ConditionNode("is_curious", _check_curiosity))
    main_seq.add_child(ActionNode("pick_inspect_target", action_fn=_pick_random_object))
    main_seq.add_child(NavigateAction("move_to_target", "navigate_target"))
    main_seq.add_child(ProgressAction("inspect_process", ACTION_DURATION["inspect"],
        state_update_fn=_inspecting_update))
    root.add_child(main_seq)

    root.add_child(Sequence("nothing_interesting", [
        ActionNode("report_bored", action_fn=lambda bb: (
            _set_flag(bb, "report_msg", "没什么有趣的"),
            BTStatus.FAILURE
        ))
    ]))

    bt = BehaviorTree(root, "curious_inspect")
    return bt


def _inspecting_update(bb: Blackboard):
    state = bb.get("state")
    if state:
        state.curiosity -= 0.08
        state.comfort += 0.02


def build_follow_player_bt() -> BehaviorTree:
    """跟随玩家移动"""
    root = Sequence("follow_main")
    root.add_child(ConditionNode("trusts_player", lambda bb: (
        bb.get("state") is not None and bb.get("state").trust_level >= 60
    )))
    root.add_child(ConditionNode("player_visible", _check_player_nearby))
    root.add_child(ActionNode("set_player_target", action_fn=lambda bb: (
        bb.set("navigate_target", bb.get("env").player_position),
        BTStatus.SUCCESS
    )))
    root.add_child(NavigateAction("follow_player_move", "navigate_target"))
    root.add_child(ProgressAction("trail_behind", ACTION_DURATION["follow"]))

    bt = BehaviorTree(root, "follow_player")
    return bt


def build_accept_petting_bt() -> BehaviorTree:
    """接受抚摸/享受互动"""
    root = Selector("petting_root")

    main_seq = Sequence("petting_main")
    main_seq.add_child(ConditionNode("player_petting", lambda bb: (
        bb.get("env") is not None and bb.get("env").player_action == "pet"
    )))
    main_seq.add_child(ConditionNode("trust_ok", lambda bb: (
        bb.get("state") is not None and bb.get("state").trust_level >= 20
    )))
    main_seq.add_child(ConditionNode("player_close", lambda bb: (
        bb.get("env") is not None and
        bb.get("env").manhattan_distance(
            bb.get("state").position, bb.get("env").player_position
        ) <= 2
    )))
    # 犹豫（根据性格参数）
    main_seq.add_child(ProgressAction("hesitate", 1))
    main_seq.add_child(ActionNode("accept_petting_action", action_fn=_petting_fn))
    main_seq.add_child(ProgressAction("enjoy_petting", ACTION_DURATION["petting"],
        state_update_fn=_petting_update))
    root.add_child(main_seq)

    root.add_child(Sequence("decline_petting", [
        ActionNode("move_away", action_fn=lambda bb: (
            _set_flag(bb, "report_msg", "还不想被摸"),
            BTStatus.FAILURE
        ))
    ]))

    bt = BehaviorTree(root, "accept_petting")
    return bt


def _petting_fn(bb: Blackboard) -> BTStatus:
    state = bb.get("state")
    if state and state.fear > 0.5:
        return BTStatus.FAILURE
    return BTStatus.SUCCESS


def _petting_update(bb: Blackboard):
    state = bb.get("state")
    if state:
        state.comfort += 5.0 / 100.0
        state.trust_level += 2.0
        state.stress_level -= 4.0


def build_fearful_retreat_bt() -> BehaviorTree:
    """恐惧后退/躲避"""
    root = Selector("retreat_root")

    main_seq = Sequence("retreat_main")
    main_seq.add_child(ConditionNode("is_fearful", lambda bb: (
        bb.get("state") is not None and bb.get("state").fear > 0.4
    )))
    # 计算远离玩家/威胁的方向
    main_seq.add_child(ActionNode("calc_retreat_dir", action_fn=_calc_retreat_target))
    main_seq.add_child(ActionNode("quick_retreat", action_fn=_quick_retreat_fn))
    main_seq.add_child(ProgressAction("watch_cautiously", ACTION_DURATION["retreat"],
        state_update_fn=_retreat_update))
    root.add_child(main_seq)

    root.add_child(Sequence("no_threat", [
        ActionNode("stand_still", action_fn=lambda bb: BTStatus.SUCCESS)
    ]))

    bt = BehaviorTree(root, "fearful_retreat")
    return bt


def _calc_retreat_target(bb: Blackboard) -> BTStatus:
    env = bb.get("env")
    state = bb.get("state")
    if env is None or state is None:
        return BTStatus.FAILURE

    player_pos = env.player_position
    my_pos = state.position
    params = bb.get("behavior_params", {})
    flee_dist = params.get("flee_distance", 3.0)

    # 远离玩家方向
    dx = my_pos[0] - player_pos[0]
    dy = my_pos[1] - player_pos[1]
    dist = max(1, (dx**2 + dy**2)**0.5)
    # 朝远离方向移动
    flee_x = int(my_pos[0] + dx / dist * flee_dist)
    flee_y = int(my_pos[1] + dy / dist * flee_dist)
    flee_x = max(0, min(env.grid.shape[1] - 1, flee_x))
    flee_y = max(0, min(env.grid.shape[0] - 1, flee_y))

    # 如果方向不可行走，找最近的躲藏点
    if not env.is_walkable((flee_x, flee_y)):
        hiding = env.get_nearest_object(my_pos, "hiding_box")
        if hiding:
            bb.set("navigate_target", hiding.position)
        else:
            bb.set("navigate_target", _random_far_pos(bb))
    else:
        bb.set("navigate_target", (flee_x, flee_y))
    return BTStatus.SUCCESS


def _quick_retreat_fn(bb: Blackboard) -> BTStatus:
    """快速后退，移动速度加倍"""
    env = bb.get("env")
    state = bb.get("state")
    target = bb.get("navigate_target")
    if env is None or state is None or target is None:
        return BTStatus.FAILURE

    # 快速移动2步
    for _ in range(2):
        new_pos = env.move_toward(state.position, target, steps=1)
        if new_pos != state.position:
            state.position = new_pos
            cat_id = bb.get("cat_id", "")
            if cat_id:
                env.cat_positions[cat_id] = new_pos
                state.current_room_id = env.get_room_id_at(new_pos)

    if env.manhattan_distance(state.position, target) <= 2:
        return BTStatus.SUCCESS
    return BTStatus.RUNNING if state.fear > 0.6 else BTStatus.SUCCESS


def _retreat_update(bb: Blackboard):
    state = bb.get("state")
    if state:
        state.fear -= 3.0 / 100.0
        state.curiosity += 1.0 / 100.0


def build_stare_at_window_bt() -> BehaviorTree:
    """看窗外/发呆"""
    root = Selector("stare_root")

    main_seq = Sequence("stare_main")
    main_seq.add_child(ConditionNode("peaceful", lambda bb: (
        bb.get("state") is not None and
        bb.get("state").fear < 0.4 and bb.get("state").stress_level < 50
    )))
    main_seq.add_child(ActionNode("find_window", action_fn=lambda bb: (
        _find_nearest_and_set_target(bb, "window_spot"),
        BTStatus.SUCCESS
    )))
    main_seq.add_child(NavigateAction("move_to_window", "navigate_target"))
    main_seq.add_child(ProgressAction("stare_outside", ACTION_DURATION["stare"],
        state_update_fn=_staring_update))
    root.add_child(main_seq)

    # 找不到窗户就原地发呆
    root.add_child(ForceSuccess("sit_and_stare", [
        ProgressAction("stare_at_nothing", ACTION_DURATION["stare"] // 2,
            state_update_fn=lambda bb: (
                bb.get("state") and _update_comfy(bb.get("state"))
            ))
    ]))

    bt = BehaviorTree(root, "stare_at_window")
    return bt


def _staring_update(bb: Blackboard):
    state = bb.get("state")
    if state:
        state.comfort += 2.0 / 100.0
        state.curiosity -= 0.02
        state.stress_level -= 1.0


def _update_comfy(state):
    state.comfort += 1.0 / 100.0


# ==================== 辅助函数 ====================

def _random_nearby_pos(bb: Blackboard) -> Tuple:
    """获取当前位置附近的随机可行走位置"""
    env = bb.get("env")
    state = bb.get("state")
    if env is None or state is None:
        return (0, 0)

    x, y = state.position
    for _ in range(20):
        nx = x + random.randint(-3, 3)
        ny = y + random.randint(-3, 3)
        if env.is_walkable((nx, ny)):
            return (nx, ny)
    return state.position


def _random_far_pos(bb: Blackboard) -> Tuple:
    """获取远离当前位置的随机位置"""
    env = bb.get("env")
    state = bb.get("state")
    if env is None or state is None:
        return (0, 0)

    x, y = state.position
    for _ in range(50):
        nx = x + random.randint(-8, 8)
        ny = y + random.randint(-8, 8)
        if env.is_walkable((nx, ny)):
            return (nx, ny)
    return state.position


def _find_nearest_and_set_target(bb: Blackboard, *obj_types: str):
    """找到最近的指定类型物体并设置为导航目标"""
    env = bb.get("env")
    state = bb.get("state")
    if env is None or state is None:
        return

    best = None
    best_dist = float('inf')
    for obj in env.objects.values():
        if obj.obj_type in obj_types:
            dist = env.manhattan_distance(state.position, obj.position)
            if dist < best_dist:
                best_dist = dist
                best = obj

    if best:
        bb.set("navigate_target", best.position)
    else:
        bb.set("navigate_target", state.position)


def _set_flag(bb: Blackboard, flag: str, value: bool):
    """设置状态标志"""
    state = bb.get("state")
    if state is None:
        return
    if flag == "set_eating":
        state.is_eating = value
    elif flag == "set_sleeping":
        state.is_sleeping = value
    elif flag == "set_hiding":
        state.is_hiding = value


def _pick_social_partner(bb: Blackboard) -> BTStatus:
    """从附近猫咪中选择社交对象"""
    nearby = bb.get("_nearby_cats", [])
    if nearby:
        bb.set("_social_target", random.choice(nearby))
        return BTStatus.SUCCESS
    return BTStatus.FAILURE


def _set_target_to_cat(bb: Blackboard, target_cat_id: str) -> BTStatus:
    """设置导航目标为指定猫咪位置"""
    env = bb.get("env")
    if env is None or not target_cat_id:
        return BTStatus.FAILURE
    pos = env.cat_positions.get(target_cat_id)
    if pos:
        bb.set("navigate_target", pos)
        return BTStatus.SUCCESS
    return BTStatus.FAILURE


def _pick_random_object(bb: Blackboard) -> BTStatus:
    """随机选择一个环境物体作为探索目标"""
    env = bb.get("env")
    state = bb.get("state")
    if env is None or state is None or not env.objects:
        return BTStatus.FAILURE
    # 优先选择玩具和未知物体
    objs = list(env.objects.values())
    weights = []
    for obj in objs:
        if obj.is_toy:
            weights.append(3.0)
        elif obj.obj_type == "scratching_post":
            weights.append(2.0)
        elif obj.obj_type == "window_spot":
            weights.append(2.5)
        else:
            weights.append(1.0)

    chosen = random.choices(objs, weights=weights, k=1)[0]
    bb.set("navigate_target", chosen.position)
    return BTStatus.SUCCESS


# ==================== 意图→行为树映射 ====================

def build_bt_for_intent(intent: str) -> BehaviorTree:
    """根据意图名称构建对应的行为树"""
    builders = {
        "idle_wander": build_idle_wander_bt,
        "approach_player": build_approach_player_bt,
        "ask_for_attention": build_ask_for_attention_bt,
        "eat": build_eat_bt,
        "sleep": build_sleep_bt,
        "play_with_toy": build_play_with_toy_bt,
        "social_groom": build_social_groom_bt,
        "social_play": build_social_play_bt,
        "hide": build_hide_bt,
        "hiss_warning": build_hiss_warning_bt,
        "curious_inspect": build_curious_inspect_bt,
        "follow_player": build_follow_player_bt,
        "accept_petting": build_accept_petting_bt,
        "fearful_retreat": build_fearful_retreat_bt,
        "stare_at_window": build_stare_at_window_bt,
    }

    builder = builders.get(intent)
    if builder is None:
        raise ValueError(f"未知意图: {intent}")
    return builder()


def get_all_behavior_trees() -> Dict[str, BehaviorTree]:
    """获取所有意图的行为树"""
    return {intent: build_bt_for_intent(intent) for intent in INTENT_LIST}
