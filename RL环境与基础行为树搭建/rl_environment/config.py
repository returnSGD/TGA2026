"""
《猫语心声》RL环境与行为树 —— 全局配置常量
基于技术策划案v2 - HRLTM架构
"""

# ==================== 宏观意图枚举 ====================
INTENT_LIST = [
    "idle_wander",       # 1  闲逛/发呆
    "approach_player",   # 2  主动靠近玩家
    "ask_for_attention", # 3  撒娇/蹭玩家/求关注
    "eat",               # 4  进食
    "sleep",             # 5  睡觉/打盹
    "play_with_toy",     # 6  玩指定玩具
    "social_groom",      # 7  与某只猫互相舔毛
    "social_play",       # 8  与某只猫玩耍/追逐
    "hide",              # 9  躲藏
    "hiss_warning",      # 10 发出警告/哈气
    "curious_inspect",   # 11 好奇探索某物/某处
    "follow_player",     # 12 跟随玩家移动
    "accept_petting",    # 13 接受抚摸/享受互动
    "fearful_retreat",   # 14 恐惧后退/躲避
    "stare_at_window",   # 15 看窗外/发呆
]
INTENT_NUM = len(INTENT_LIST)

# ==================== 性格维度定义 ====================
PERSONALITY_KEYS = ["傲娇", "怯懦", "好奇", "贪吃", "社交", "攻击性", "活跃", "独立"]
PERSONALITY_DIM = 8

# ==================== 状态向量维度 ====================
EMOTION_DIM = 5         # 饥饿, 恐惧, 好奇, 舒适, 社交需求
PHYSICAL_DIM = 3        # 精力, 健康, 体温
ENV_FEATURE_DIM = 5     # 区域舒适度, 刺激度, 卫生度, 光照, 噪音
RELATION_DIM = 4        # 对玩家好感, 亲密度均值, 敌意度均值, 社交排名
PLAYER_ACTION_DIM = 12
MEMORY_EMBED_DIM = 128
TOP_K_MEMORIES = 3

STATE_DIM = (PERSONALITY_DIM + EMOTION_DIM + PHYSICAL_DIM + 1 +
             ENV_FEATURE_DIM + RELATION_DIM + PLAYER_ACTION_DIM +
             MEMORY_EMBED_DIM * TOP_K_MEMORIES)

# ==================== 记忆系统 ====================
WORK_MEMORY_SIZE = 20
LONG_MEMORY_CAP = 500
MEMORY_IMPORTANCE_THRESHOLD = 4.0
SEQ_LEN = 4

# ==================== 猫咪基础数值 ====================
# 需求衰减速率（每游戏tick，1 tick ≈ 游戏中10分钟）
HUNGER_DECAY = 0.8       # 饥饿值每tick上升
ENERGY_DECAY = 1.2       # 精力值每tick下降
COMFORT_DECAY = 0.3      # 舒适度自然衰减
SOCIAL_DECAY = 0.4       # 社交需求自然上升
STRESS_NATURAL_INCREASE = 0.2  # 压力自然上升

# 需求上下限
NEED_MIN = 0.0
NEED_MAX = 100.0
TRUST_MIN = 0.0
TRUST_MAX = 100.0

# ==================== 沙盒环境 ====================
# 网格地图
GRID_WIDTH = 20
GRID_HEIGHT = 15

# 房间定义
ROOMS = {
    "大厅":      {"x": 0,  "y": 0,  "w": 12, "h": 10, "comfort": 0.6, "stimulation": 0.4, "hygiene": 0.7},
    "后院":      {"x": 12, "y": 0,  "w": 8,  "h": 6,  "comfort": 0.8, "stimulation": 0.3, "hygiene": 0.5},
    "静音隔间":  {"x": 12, "y": 6,  "w": 5,  "h": 4,  "comfort": 0.9, "stimulation": 0.1, "hygiene": 0.8},
    "阳光温室":  {"x": 0,  "y": 10, "w": 8,  "h": 5,  "comfort": 0.7, "stimulation": 0.5, "hygiene": 0.6},
    "隔离区":    {"x": 17, "y": 6,  "w": 3,  "h": 4,  "comfort": 0.4, "stimulation": 0.2, "hygiene": 0.9},
}

# 物体类型
OBJECT_TYPES = [
    "food_bowl",      # 食盆
    "water_bowl",     # 水盆
    "cat_bed",        # 猫窝
    "toy_mouse",      # 玩具老鼠
    "toy_ball",       # 玩具球
    "scratching_post",# 猫抓板
    "hiding_box",     # 躲藏纸箱
    "window_spot",    # 窗边位置
    "player",         # 玩家位置
]

# 游戏时间
TICKS_PER_DAY = 144    # 游戏中每天144个tick（每tick=10分钟）
DAY_PHASES = {
    "morning":   (0, 36),    # 清晨准备 6:00-12:00
    "afternoon": (36, 72),   # 下午营业 12:00-18:00
    "evening":   (72, 108),  # 傍晚 18:00-24:00
    "night":     (108, 144), # 深夜 0:00-6:00
}

# ==================== 行为树节点状态 ====================
class BTStatus:
    SUCCESS = 0
    FAILURE = 1
    RUNNING = 2

# ==================== 动作时长（tick数） ====================
ACTION_DURATION = {
    "move": 2,
    "eat": 5,
    "sleep": 15,
    "play": 4,
    "hide": 1,
    "groom": 6,
    "hiss": 1,
    "inspect": 3,
    "petting": 5,
    "stare": 8,
    "approach": 3,
    "follow": 2,
    "retreat": 2,
    "ask_attention": 3,
}

# ==================== 奖励函数参数（用于后续RL训练） ====================
REWARDS = {
    "eat_success": 0.5,
    "sleep_success": 0.3,
    "player_interact": 1.5,
    "social_positive": 0.8,
    "hide_success": 0.5,
    "approach_accepted": 1.5,
    "trust_milestone": 5.0,
    "destroy_furniture": -0.3,
    "bully_other_cat": -1.0,
    "ignore_player_daily": -0.1,
    "exploration_bonus": 0.2,
}

# ==================== 性格→意图兼容矩阵 ====================
# 值 > 0 表示该性格倾向该意图，< 0 表示排斥
INTENT_PERSONALITY_MATRIX = {
    "傲娇": {
        "approach_player": -2.0, "ask_for_attention": -3.0, "accept_petting": -2.0,
        "hiss_warning": 3.0, "hide": -1.0, "social_groom": -2.0,
        "social_play": -1.0, "follow_player": -2.0, "fearful_retreat": -1.0,
    },
    "怯懦": {
        "approach_player": -5.0, "ask_for_attention": -4.0, "hide": 4.0,
        "hiss_warning": 2.0, "curious_inspect": -2.0, "fearful_retreat": 4.0,
        "social_play": -2.0, "follow_player": -3.0, "accept_petting": -1.0,
    },
    "好奇": {
        "curious_inspect": 4.0, "approach_player": 1.0, "play_with_toy": 2.0,
        "hide": -2.0, "social_play": 1.5, "stare_at_window": 1.0,
        "follow_player": 1.0,
    },
    "贪吃": {
        "eat": 3.0, "approach_player": 0.5, "curious_inspect": 1.0,
        "idle_wander": -0.5, "sleep": -1.0,
    },
    "社交": {
        "social_groom": 4.0, "social_play": 3.0, "ask_for_attention": 3.0,
        "approach_player": 2.0, "follow_player": 2.0, "hide": -2.0,
        "hiss_warning": -2.0, "fearful_retreat": -2.0,
    },
    "攻击性": {
        "hiss_warning": 4.0, "social_groom": -3.0, "social_play": -2.0,
        "approach_player": -2.0, "ask_for_attention": -3.0, "hide": -3.0,
    },
    "活跃": {
        "social_play": 3.0, "play_with_toy": 3.0, "curious_inspect": 2.0,
        "follow_player": 1.5, "sleep": -2.0, "idle_wander": -2.0,
        "stare_at_window": -1.0,
    },
    "独立": {
        "idle_wander": 2.0, "stare_at_window": 2.0, "hide": 1.0,
        "ask_for_attention": -3.0, "follow_player": -3.0, "social_groom": -2.0,
        "approach_player": -1.5,
    },
}

# ==================== 性格→行为参数表 ====================
PERSONALITY_BEHAVIOR_PARAMS = {
    "傲娇": {"approach_distance": 1.5, "move_speed": 0.5, "response_delay": 2.5,
             "flee_distance": 1.5, "hesitation_weight": 0.6},
    "怯懦": {"approach_distance": 3.0, "move_speed": 0.3, "response_delay": 6.0,
             "flee_distance": 4.0, "hesitation_weight": 0.9},
    "好奇": {"approach_distance": 1.0, "move_speed": 0.7, "response_delay": 1.0,
             "flee_distance": 2.0, "hesitation_weight": 0.2},
    "贪吃": {"approach_distance": 0.5, "move_speed": 0.8, "response_delay": 0.5,
             "flee_distance": 1.0, "hesitation_weight": 0.1},
    "社交": {"approach_distance": 0.5, "move_speed": 0.8, "response_delay": 0.5,
             "flee_distance": 1.0, "hesitation_weight": 0.1},
    "攻击性": {"approach_distance": 0.8, "move_speed": 0.9, "response_delay": 0.3,
              "flee_distance": 0.5, "hesitation_weight": 0.0},
    "活跃": {"approach_distance": 0.6, "move_speed": 1.0, "response_delay": 0.3,
             "flee_distance": 1.0, "hesitation_weight": 0.1},
    "独立": {"approach_distance": 2.0, "move_speed": 0.5, "response_delay": 1.5,
             "flee_distance": 2.0, "hesitation_weight": 0.4},
}

# ==================== 三只初始猫咪配置 ====================
CAT_CONFIGS = {
    "xiaoxue": {
        "name": "小雪",
        "breed": "纯白波斯猫",
        "personality": [0.0, 0.9, 0.1, 0.2, 0.1, 0.0, 0.1, 0.3],  # 傲娇,怯懦,好奇,贪吃,社交,攻击性,活跃,独立
        "backstory": "曾被前任主人锁在黑暗航空箱内遗弃在雨中，遭受严重心理创伤。",
        "traits": "怯懦, 敏感社恐, 缺乏安全感, 容易受惊吓, 极其温柔",
        "cat_type": "stray",
        "trust_init": 5,
        "stress_init": 85,
        "color": "white",
    },
    "oreo": {
        "name": "奥利奥",
        "breed": "奶牛猫",
        "personality": [0.8, 0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.7],  # 傲娇,怯懦,好奇,贪吃,社交,攻击性,活跃,独立
        "backstory": "曾是街头群落的霸主，为保护食物与其他野猫搏斗过。",
        "traits": "傲娇, 领地意识强, 表面高冷实则渴望关注, 老大哥做派",
        "cat_type": "native",
        "trust_init": 30,
        "stress_init": 40,
        "color": "black_white",
    },
    "orange": {
        "name": "橘子",
        "breed": "橘猫（幼年）",
        "personality": [0.0, 0.0, 0.8, 0.9, 0.7, 0.0, 0.8, 0.1],  # 贪吃, 好奇, 乐观, 活泼
        "backstory": "刚出生不久在暴雨天被玩家在垃圾桶旁发现，未经历深度创伤。",
        "traits": "贪吃, 无忧无虑, 极其乐观, 缺乏边界感, 纯粹的乐天派",
        "cat_type": "native",
        "trust_init": 60,
        "stress_init": 15,
        "color": "orange",
    },
}
