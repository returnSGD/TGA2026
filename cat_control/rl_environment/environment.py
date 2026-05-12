"""
《猫语心声》 —— 沙盒模拟环境

2D网格地图，模拟猫咖空间布局。
包含：房间、物体、猫咪移动、需求衰减、玩家交互。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import math
import random
import numpy as np

from .config import (
    GRID_WIDTH, GRID_HEIGHT, ROOMS, OBJECT_TYPES,
    DAY_PHASES, TICKS_PER_DAY,
    HUNGER_DECAY, ENERGY_DECAY, COMFORT_DECAY, SOCIAL_DECAY,
    STRESS_NATURAL_INCREASE, NEED_MIN, NEED_MAX,
)


@dataclass
class EnvObject:
    """环境中的物体"""
    obj_type: str
    position: Tuple[int, int]
    room_id: int = 0
    durability: float = 1.0
    data: Dict = field(default_factory=dict)

    @property
    def is_food_bowl(self) -> bool:
        return self.obj_type == "food_bowl"

    @property
    def is_bed(self) -> bool:
        return self.obj_type == "cat_bed"

    @property
    def is_hiding_spot(self) -> bool:
        return self.obj_type == "hiding_box"

    @property
    def is_toy(self) -> bool:
        return self.obj_type in ("toy_mouse", "toy_ball")


class SandboxEnvironment:
    """
    猫咪模拟沙盒 —— 无UI纯逻辑层

    管理：地图、物体、猫咪位置、时间推进、环境状态
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

        # 2D网格（0=空地, 1=墙壁, 2=障碍物）
        self.grid: np.ndarray = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)
        self._init_map_layout()

        # 物体映射
        self.objects: Dict[int, EnvObject] = {}
        self._obj_id_counter = 0
        self._init_default_objects()

        # 房间信息
        self.room_env = {name: {
            "comfort": info["comfort"],
            "stimulation": info["stimulation"],
            "hygiene": info["hygiene"],
            "light": 0.5,
            "noise": 0.3,
        } for name, info in ROOMS.items()}

        # 时间系统
        self.game_tick: int = 0
        self.day_count: int = 1
        self.day_phase: str = "morning"
        self.weather: str = "晴"

        # 玩家状态
        self.player_position: Tuple[int, int] = (6, 5)
        self.player_action: str = "none"
        self.player_in_cat_ear: bool = False

        # 猫咪位置记录
        self.cat_positions: Dict[str, Tuple[int, int]] = {}

        # 事件日志
        self.event_log: List[str] = []

        # 邻居缓存
        self._neighbor_cache: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    def _init_map_layout(self):
        """初始化简易地图布局"""
        # 设置墙壁边界
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        # 大厅内部无障碍
        # 后院与大厅有通道
        # 静音隔间在小角落
        self._add_wall(12, 5, 12, 9)  # 隔间隔墙

        # 为静音隔间开门
        self.grid[9, 12] = 0

    def _add_wall(self, x1: int, y1: int, x2: int, y2: int):
        y1, y2 = min(y1, y2), max(y1, y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:
                    self.grid[y, x] = 1

    def _init_default_objects(self):
        """初始化默认家具和物品"""
        default_objects = [
            # 大厅设施
            ("food_bowl", (4, 3), "大厅"),
            ("food_bowl", (7, 3), "大厅"),
            ("water_bowl", (5, 4), "大厅"),
            ("cat_bed", (3, 7), "大厅"),
            ("cat_bed", (8, 6), "大厅"),
            ("cat_bed", (10, 4), "大厅"),
            ("scratching_post", (2, 2), "大厅"),
            ("scratching_post", (9, 2), "大厅"),
            ("toy_mouse", (6, 8), "大厅"),
            ("toy_ball", (4, 6), "大厅"),
            ("hiding_box", (1, 3), "大厅"),
            ("hiding_box", (10, 7), "大厅"),
            ("window_spot", (1, 1), "大厅"),

            # 后院设施
            ("food_bowl", (14, 3), "后院"),
            ("cat_bed", (16, 4), "后院"),
            ("scratching_post", (15, 2), "后院"),
            ("toy_ball", (13, 4), "后院"),
            ("hiding_box", (18, 2), "后院"),

            # 静音隔间
            ("food_bowl", (13, 7), "静音隔间"),
            ("cat_bed", (14, 8), "静音隔间"),
            ("hiding_box", (15, 8), "静音隔间"),

            # 阳光温室
            ("cat_bed", (3, 11), "阳光温室"),
            ("window_spot", (1, 11), "阳光温室"),
            ("toy_mouse", (5, 13), "阳光温室"),
        ]

        for obj_type, pos, room_name in default_objects:
            room_id = list(ROOMS.keys()).index(room_name)
            self.add_object(obj_type, pos, room_id)

    def add_object(self, obj_type: str, pos: Tuple[int, int],
                   room_id: int = 0) -> int:
        obj_id = self._obj_id_counter
        self._obj_id_counter += 1
        self.objects[obj_id] = EnvObject(
            obj_type=obj_type, position=pos, room_id=room_id
        )
        return obj_id

    def get_room_id_at(self, pos: Tuple[int, int]) -> int:
        """获取指定位置所属房间ID"""
        x, y = pos
        for i, (name, info) in enumerate(ROOMS.items()):
            rx, ry, rw, rh = info["x"], info["y"], info["w"], info["h"]
            if rx <= x < rx + rw and ry <= y < ry + rh:
                return i
        return 0  # 默认大厅

    def get_room_name(self, room_id: int) -> str:
        return list(ROOMS.keys())[room_id] if 0 <= room_id < len(ROOMS) else "未知"

    def get_room_env(self, room_id: int) -> Dict[str, float]:
        name = self.get_room_name(room_id)
        return self.room_env.get(name, {
            "comfort": 0.5, "stimulation": 0.3, "hygiene": 0.5,
            "light": 0.5, "noise": 0.3,
        })

    def get_objects_in_room(self, room_id: int) -> List[EnvObject]:
        return [obj for obj in self.objects.values() if obj.room_id == room_id]

    def get_objects_at(self, pos: Tuple[int, int]) -> List[EnvObject]:
        return [obj for obj in self.objects.values() if obj.position == pos]

    def get_nearest_object(self, pos: Tuple[int, int],
                           obj_type: str) -> Optional[EnvObject]:
        """获取最近的指定类型物体"""
        candidates = [obj for obj in self.objects.values() if obj.obj_type == obj_type]
        if not candidates:
            return None
        return min(candidates, key=lambda o: self.manhattan_distance(pos, o.position))

    def get_nearby_objects(self, pos: Tuple[int, int],
                           max_dist: int = 2) -> List[EnvObject]:
        """获取指定位置附近的物体"""
        result = []
        for obj in self.objects.values():
            if self.manhattan_distance(pos, obj.position) <= max_dist:
                result.append(obj)
        return result

    def get_nearby_cats(self, cat_id: str, max_dist: int = 3) -> List[str]:
        """获取附近的猫咪ID列表"""
        my_pos = self.cat_positions.get(cat_id)
        if my_pos is None:
            return []
        result = []
        for cid, pos in self.cat_positions.items():
            if cid != cat_id and self.manhattan_distance(my_pos, pos) <= max_dist:
                result.append(cid)
        return result

    def manhattan_distance(self, p1: Tuple[int, int],
                           p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def is_walkable(self, pos: Tuple[int, int]) -> bool:
        """判断位置是否可行走"""
        x, y = int(pos[0]), int(pos[1])
        if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
            return False
        return self.grid[y, x] == 0

    def find_path(self, start: Tuple[int, int],
                  goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """简易BFS寻路"""
        if start == goal:
            return [start]
        if not self.is_walkable(goal):
            return []

        from collections import deque
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            (cx, cy), path = queue.popleft()
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                np_pos = (nx, ny)
                if np_pos == goal:
                    return path + [np_pos]
                if np_pos not in visited and self.is_walkable(np_pos):
                    visited.add(np_pos)
                    queue.append((np_pos, path + [np_pos]))
        return []  # 无路径

    def move_toward(self, current: Tuple[int, int],
                    target: Tuple[int, int], steps: int = 1) -> Tuple[int, int]:
        """向目标移动steps步"""
        path = self.find_path(current, target)
        if not path:
            return current
        idx = min(steps, len(path) - 1)
        new_pos = path[idx]
        if self.is_walkable(new_pos):
            return new_pos
        return current

    # ==================== 时间与需求系统 ====================

    def advance_tick(self):
        """推进一个游戏tick"""
        self.game_tick += 1

        # 更新阶段
        tick_of_day = self.game_tick % TICKS_PER_DAY
        for phase, (start, end) in DAY_PHASES.items():
            if start <= tick_of_day < end:
                self.day_phase = phase
                break

        # 新一天
        if tick_of_day == 0 and self.game_tick > 0:
            self.day_count += 1
            self._on_new_day()

        # 更新时间相关环境参数
        self._update_environment()

    def _update_environment(self):
        """随时间更新环境参数"""
        tick_of_day = self.game_tick % TICKS_PER_DAY

        # 光照：白天高、夜晚低
        if 18 <= tick_of_day < 90:  # 白天
            light_base = 0.7 + 0.3 * math.sin((tick_of_day - 18) / 72 * math.pi)
        elif 90 <= tick_of_day < 126:  # 傍晚
            light_base = max(0.1, 0.7 - (tick_of_day - 90) / 36 * 0.6)
        else:
            light_base = 0.1

        # 噪音：营业时间高
        if 36 <= tick_of_day < 108:
            noise_base = 0.4 + random.uniform(-0.1, 0.2)
        else:
            noise_base = 0.1 + random.uniform(-0.05, 0.05)

        for room_name in self.room_env:
            self.room_env[room_name]["light"] = light_base
            self.room_env[room_name]["noise"] = noise_base

        # 随机天气变化
        if random.random() < 0.02:
            weathers = ["晴", "多云", "小雨", "阵雨"]
            self.weather = random.choice(weathers)

    def _on_new_day(self):
        """新一天触发"""
        self.log_event(f"=== 第{self.day_count}天开始，天气: {self.weather} ===")

    def apply_need_decay(self, state: 'CatState'):
        """
        对猫咪状态应用需求衰减。

        饥饿上升、精力下降、舒适度自然衰减、社交需求上升、压力自然微增。
        环境因素和物体影响由外部调用。
        """
        state.hunger += HUNGER_DECAY / 100.0
        state.energy -= ENERGY_DECAY / 100.0
        state.comfort -= COMFORT_DECAY / 100.0
        state.social_need += SOCIAL_DECAY / 100.0
        state.stress_level += STRESS_NATURAL_INCREASE

        # 睡眠时特殊处理：精力恢复、舒适提升
        if state.is_sleeping:
            state.energy += 5.0 / 100.0
            state.comfort += 2.0 / 100.0
            state.stress_level -= 1.0
            state.hunger += 0.2 / 100.0  # 睡眠也消耗一点能量

        # 进食时特殊处理
        if state.is_eating:
            state.hunger -= 12.0 / 100.0

        # 躲藏时：舒适略降、恐惧上升减慢
        if state.is_hiding:
            state.comfort -= 1.0 / 100.0

        # 环境因素修正
        room_env = self.get_room_env(state.current_room_id)
        state.comfort += room_env["comfort"] * 0.5 / 100.0
        state.curiosity += room_env["stimulation"] * 0.3 / 100.0
        state.stress_level += room_env["noise"] * 2.0

        # 高恐惧加速压力
        if state.fear > 0.7:
            state.stress_level += 1.5

        # 社交需求补偿：附近有猫则下降
        if state.nearby_cats:
            state.social_need -= 2.0 / 100.0

    def set_player_action(self, action: str, target: Optional[Tuple[int, int]] = None):
        self.player_action = action
        if target:
            self.player_position = target

    def log_event(self, msg: str):
        self.event_log.append(f"[tick {self.game_tick}] {msg}")
        if len(self.event_log) > 500:
            self.event_log = self.event_log[-500:]

    def get_recent_events(self, n: int = 10) -> List[str]:
        return self.event_log[-n:]

    def environment_summary(self) -> str:
        room_name = self.get_room_name(0)
        env = self.room_env.get(room_name, {})
        return (
            f"Day {self.day_count} | Phase: {self.day_phase} | "
            f"Weather: {self.weather} | "
            f"Light: {env.get('light', 0):.2f} | Noise: {env.get('noise', 0):.2f}"
        )

    def grid_snapshot(self) -> np.ndarray:
        """返回当前地图快照（复制）"""
        return self.grid.copy()

    def get_state(self) -> Dict:
        """获取环境状态摘要（用于日志/调试）"""
        return {
            "tick": self.game_tick,
            "day": self.day_count,
            "phase": self.day_phase,
            "weather": self.weather,
            "player_pos": self.player_position,
            "player_action": self.player_action,
            "cat_positions": dict(self.cat_positions),
            "object_count": len(self.objects),
        }
