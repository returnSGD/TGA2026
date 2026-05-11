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

# ==================== 性格→意图兼容矩阵（完整 8×15 = 120 值） ====================
# 值 > 0 表示该性格倾向该意图，< 0 表示排斥，0 表示不影响
# 偏置范围: -5.0 到 +5.0
# 在 RL 策略网络输出 logits 后、softmax 前施加偏置：
#   filtered_logits[i] = raw_logits[i] + Σ(性格权重[j] × 矩阵[j][i])
INTENT_PERSONALITY_MATRIX = {
    # ── 傲娇：高贵冷艳，表面嫌弃实则在意 ──
    "傲娇": {
        "idle_wander":        0.5,   # 有时会独自"巡视领地"
        "approach_player":   -2.0,   # 很少主动靠近，等玩家来找它
        "ask_for_attention": -3.0,   # 绝不主动撒娇
        "eat":                0.0,   # 正常进食
        "sleep":              0.5,   # 喜欢在显眼处打盹
        "play_with_toy":     -1.0,   # 对玩具表现不屑
        "social_groom":      -2.0,   # 不愿与其他猫过于亲密
        "social_play":       -1.0,   # 偶尔参与但不热衷
        "hide":              -1.0,   # 不轻易躲藏
        "hiss_warning":       3.0,   # 不爽就哈气
        "curious_inspect":    0.0,   # 正常好奇
        "follow_player":     -2.0,   # 不跟人
        "accept_petting":    -2.0,   # 被摸时表面不情愿
        "fearful_retreat":   -1.0,   # 很少逃跑
        "stare_at_window":    1.0,   # 远眺沉思
    },
    # ── 怯懦：胆小敏感，缺乏安全感 ──
    "怯懦": {
        "idle_wander":        0.0,
        "approach_player":   -5.0,   # 极难主动靠近人类
        "ask_for_attention": -4.0,   # 完全不敢撒娇
        "eat":                0.5,   # 偷偷进食
        "sleep":             -0.5,   # 需要极度安全才敢睡
        "play_with_toy":     -1.5,   # 害怕玩具
        "social_groom":      -1.0,   # 社交谨慎
        "social_play":       -2.0,   # 不敢参与追逐
        "hide":               4.0,   # 擅长躲藏
        "hiss_warning":       2.0,   # 极端恐惧时哈气
        "curious_inspect":   -2.0,   # 对新鲜事物充满警惕
        "follow_player":     -3.0,   # 不敢跟随
        "accept_petting":    -1.0,   # 需要极长时间才接受
        "fearful_retreat":    4.0,   # 稍有风吹草动就撤退
        "stare_at_window":    1.0,   # 安静远望
    },
    # ── 好奇：探索欲强，什么都想碰 ──
    "好奇": {
        "idle_wander":        0.5,
        "approach_player":    1.0,   # 好奇玩家的行为
        "ask_for_attention":  0.5,   # 偶尔因为好奇而互动
        "eat":                0.0,
        "sleep":             -1.0,   # 不爱睡觉，想玩
        "play_with_toy":      2.0,   # 对新玩具充满兴趣
        "social_groom":       0.5,
        "social_play":        1.5,   # 愿意与其他猫探索
        "hide":              -2.0,   # 不太躲藏
        "hiss_warning":      -1.0,
        "curious_inspect":    4.0,   # 探索行为的核心驱动力
        "follow_player":      1.0,   # 好奇玩家去哪
        "accept_petting":     0.5,   # 好奇抚摸的感觉
        "fearful_retreat":   -1.5,   # 好奇心压过恐惧
        "stare_at_window":    1.0,   # 观察窗外世界
    },
    # ── 贪吃：食欲旺盛，以食为天 ──
    "贪吃": {
        "idle_wander":       -0.5,
        "approach_player":    0.5,   # 为食物靠近玩家
        "ask_for_attention":  1.0,   # 用撒娇换取食物
        "eat":                3.0,   # 进食第一优先级
        "sleep":             -1.0,   # 少睡多吃
        "play_with_toy":     -0.5,   # 除非玩具藏了食物
        "social_groom":       0.0,
        "social_play":        0.0,
        "hide":              -1.0,
        "hiss_warning":      -0.5,   # 护食时可能哈气
        "curious_inspect":    1.0,   # 到处找吃的
        "follow_player":      0.5,   # 跟着玩家=可能有零食
        "accept_petting":     1.0,   # 吃东西时被摸无所谓
        "fearful_retreat":   -0.5,
        "stare_at_window":   -1.0,
    },
    # ── 社交：友好亲昵，喜欢陪伴 ──
    "社交": {
        "idle_wander":        0.0,
        "approach_player":    2.0,   # 喜欢靠近人
        "ask_for_attention":  3.0,   # 主动撒娇求关注
        "eat":                0.0,
        "sleep":              0.0,
        "play_with_toy":      1.0,   # 愿意和人/猫一起玩
        "social_groom":       4.0,   # 互相舔毛是爱的表达
        "social_play":        3.0,   # 喜欢追逐游戏
        "hide":              -2.0,   # 不爱躲藏
        "hiss_warning":      -2.0,   # 很少哈气
        "curious_inspect":    0.5,
        "follow_player":      2.0,   # 喜欢跟着人
        "accept_petting":     2.0,   # 享受抚摸
        "fearful_retreat":   -2.0,   # 很少逃跑
        "stare_at_window":    0.0,
    },
    # ── 攻击性：易怒护领地，不容侵犯 ──
    "攻击性": {
        "idle_wander":        0.0,
        "approach_player":   -2.0,   # 不愿靠近
        "ask_for_attention": -3.0,   # 不可能撒娇
        "eat":                0.0,
        "sleep":              0.0,
        "play_with_toy":     -1.0,
        "social_groom":      -3.0,   # 厌恶亲密接触
        "social_play":       -2.0,   # 玩耍容易变真打
        "hide":              -3.0,   # 不躲藏，正面迎击
        "hiss_warning":       4.0,   # 哈气警告是本能
        "curious_inspect":   -0.5,
        "follow_player":     -2.5,
        "accept_petting":    -4.0,   # 极难接受抚摸
        "fearful_retreat":   -3.0,   # 宁可战斗也不逃跑
        "stare_at_window":    0.5,
    },
    # ── 活跃：精力充沛，停不下来 ──
    "活跃": {
        "idle_wander":       -2.0,   # 静不下来
        "approach_player":    1.0,
        "ask_for_attention":  1.0,   # 用行动吸引注意
        "eat":                0.0,
        "sleep":             -2.0,   # 不爱睡觉
        "play_with_toy":      3.0,   # 玩具消耗精力
        "social_groom":       0.0,
        "social_play":        3.0,   # 追逐打闹是最爱
        "hide":              -1.5,
        "hiss_warning":       0.0,
        "curious_inspect":    2.0,   # 到处探索
        "follow_player":      1.5,   # 跟随移动
        "accept_petting":     1.0,   # 短暂接受后继续动
        "fearful_retreat":   -1.0,
        "stare_at_window":   -1.0,
    },
    # ── 独立：我行我素，不需要人 ──
    "独立": {
        "idle_wander":        2.0,   # 喜欢自己闲逛
        "approach_player":   -1.5,   # 不太主动靠近
        "ask_for_attention": -3.0,   # 不需要撒娇
        "eat":                0.5,   # 自己找吃的
        "sleep":              1.0,   # 独自安睡
        "play_with_toy":      0.0,   # 自娱自乐
        "social_groom":      -2.0,   # 不追求社交
        "social_play":       -1.0,
        "hide":               1.0,   # 独自躲藏
        "hiss_warning":       1.0,   # 不喜欢被打扰
        "curious_inspect":    1.0,   # 独立探索
        "follow_player":     -3.0,   # 绝不跟人
        "accept_petting":    -1.5,   # 不太喜欢被摸
        "fearful_retreat":    0.5,
        "stare_at_window":    2.0,   # 最爱独处远眺
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

# ==================== 性格关键词禁止列表 ====================
# 第三层过滤器：LLM 生成心声文本后检查，若含禁止词则回退到模板库
# 性格向量某维度 > 0.7 时，该维度的所有禁用词生效
PERSONALITY_FORBIDDEN_WORDS = {
    "傲娇": [
        "最喜欢你了", "我好想你", "陪我玩嘛", "好开心",
        "太棒了", "好兴奋", "爱你哦", "抱抱", "蹭蹭你",
        "快点过来", "不要走嘛", "呜呜", "求你了",
    ],
    "怯懦": [
        "最喜欢", "主动过来", "我不怕", "陪我玩", "快点",
        "过来啊", "有什么好怕的", "我才不怕", "看我的",
        "冲上去", "好勇敢", "放马过来",
    ],
    "好奇": [
        # 好奇猫限制较少，但防止过度大胆
        "我一点都不好奇", "没意思", "无聊",
    ],
    "贪吃": [
        # 贪吃猫限制较少
        "我不想吃", "不饿", "没胃口", "吃腻了",
    ],
    "社交": [
        # 社交猫限制较少，防止社交排斥言论
        "走开", "别碰我", "烦死了", "别过来",
    ],
    "攻击性": [
        "爱你", "蹭蹭", "抱抱", "最喜欢", "好温柔",
        "摸摸我", "你好暖", "舒服", "再来一次",
        "我好幸福", "好甜蜜",
    ],
    "活跃": [
        # 活跃猫限制较少
        "好累", "不想动", "没力气", "懒得",
    ],
    "独立": [
        "陪陪我", "不要走", "好孤单", "我想你",
        "别离开", "一个人好害怕", "谁来陪陪我",
        "抱紧我", "贴贴", "好需要你",
    ],
}

# ==================== 事件重要性评分表 ====================
# 用于长期记忆系统：每条记忆的 base_importance 由事件类型决定
# score = base_importance × time_decay
# time_decay = max(0.1, 1 - (t_now - t_event) / max_ttl)
EVENT_IMPORTANCE_BASE = {
    # ── 里程碑级 (9.0 ~ 10.0)：一生只发生几次 ──
    "milestone_trust_breakthrough":     (9.0, 10.0),  # 信任突破阈值（20/40/65/90）
    "milestone_first_approach":         (9.0, 10.0),  # 流浪猫第一次主动靠近玩家
    "milestone_first_purr":             (9.0, 10.0),  # 第一次发出呼噜声
    "milestone_adoption_complete":      (9.5, 10.0),  # 完成领养/永居选择

    # ── 关键情感事件 (8.0 ~ 9.0) ──
    "first_pet_accepted":               (8.0, 9.0),   # 第一次接受抚摸
    "first_treat_from_hand":            (8.0, 9.0),   # 第一次从玩家手中吃零食
    "first_voluntary_rub":              (8.5, 9.5),   # 第一次主动蹭玩家
    "first_night_sleep_near_player":    (8.0, 8.5),   # 第一次在玩家附近入睡

    # ── 深度情感互动 (7.0 ~ 8.0) ──
    "deep_healing_session":             (7.0, 8.0),   # 深度心理疗愈事件
    "trauma_trigger_resolved":          (7.5, 8.5),   # 创伤触发后被成功安抚
    "player_kept_promise":              (7.0, 8.0),   # 玩家履行了之前的约定

    # ── 猫咪社交关键事件 (6.0 ~ 7.0) ──
    "social_bond_formed":               (6.0, 7.0),   # 两只敌对猫首次和解/结盟
    "first_grooming_together":          (6.0, 7.0),   # 首次互相舔毛
    "first_sleep_together":             (6.5, 7.5),   # 首次依偎入睡

    # ── 玩家特别关怀 (5.0 ~ 6.0) ──
    "player_gift":                      (5.0, 6.0),   # 玩家赠送礼物（新玩具/窝）
    "player_special_treat":             (5.0, 6.0),   # 高级零食/猫薄荷
    "player_arranged_comfort":          (5.0, 6.0),   # 玩家根据心声调整环境

    # ── 日常正面互动 (3.0 ~ 5.0) ──
    "daily_feed":                       (3.0, 5.0),   # 日常喂食
    "routine_pet_accepted":             (3.5, 5.0),   # 常规抚摸被接受
    "played_with_player":               (3.5, 5.0),   # 与玩家互动玩耍
    "player_soothe_success":            (4.0, 5.5),   # 玩家安抚成功

    # ── 应激/负面事件 (3.0 ~ 6.0) ──
    "trauma_triggered":                 (5.0, 7.0),   # 创伤被触发（需记得才能避免）
    "fight_with_other_cat":             (4.0, 6.0),   # 与其他猫打架
    "scared_by_player":                 (5.0, 7.0),   # 被玩家惊吓
    "loud_noise_scare":                 (3.0, 5.0),   # 被巨响惊吓

    # ── 日常行为 (1.0 ~ 3.0) ──
    "idle_wander":                      (1.0, 2.0),   # 闲逛
    "routine_sleep":                    (1.0, 2.0),   # 日常睡觉
    "stare_window":                     (1.0, 2.5),   # 看窗外
    "routine_explore":                  (1.5, 3.0),   # 日常探索
}

# 事件类型到基础重要性中值的快速映射（用于代码中快速查询）
EVENT_IMPORTANCE_MID = {
    k: (lo + hi) / 2.0 for k, (lo, hi) in EVENT_IMPORTANCE_BASE.items()
}

# ==================== 意图→行为树子图定义文档 ====================
# 每个宏观意图在 bt_intents.py 中对应的行为树结构描述。
# 行为树引擎确保 RL 输出的意图被翻译为安全的原子动作序列。
BT_INTENT_DESCRIPTIONS = {
    "idle_wander": {
        "description": "闲逛/发呆 — 随机选择一个可行走方向移动1-3格后短暂停留",
        "tree_type": "Selector",
        "main_path": "Sequence[随机选方向 → Navigate移动 → ProgressAction发呆2-4tick]",
        "fallback": "连续失败≥3次 → ForceSuccess[强制发呆]",
        "fail_behavior": "降级为原地停留",
        "key_conditions": ["连续失败检查"],
    },
    "approach_player": {
        "description": "主动靠近玩家 — 寻路至玩家附近，减速后等待回应",
        "tree_type": "Sequence",
        "main_path": "Sequence[玩家在场检查 → 信任足够检查 → 设目标为玩家位置 → Navigate寻路 → 减速靠近 → 等待回应3tick]",
        "fallback": "无",
        "fail_behavior": "停止靠近，维持当前距离",
        "key_conditions": ["玩家在附近(≤3格)", "信任度≥阈值"],
    },
    "ask_for_attention": {
        "description": "撒娇/蹭玩家/求关注 — 高信任时主动蹭人",
        "tree_type": "Sequence",
        "main_path": "Sequence[高信任检查(≥50) → 玩家在附近 → 移动到玩家 → 蹭人动作 → 等待反应3tick]",
        "fallback": "无",
        "fail_behavior": "放弃撒娇，退为idle_wander",
        "key_conditions": ["信任度≥50", "玩家在附近≤3格"],
    },
    "eat": {
        "description": "进食 — 寻路至最近食盆，持续进食5tick",
        "tree_type": "Selector",
        "main_path": "Sequence[饥饿>50%检查 → 食物可用检查 → 找最近食盆 → Navigate移动 → 开始进食标记 → ProgressAction进食5tick → 完成进食]",
        "fallback": "Sequence[无食物 → 报FAILURE]",
        "fail_behavior": "报告找不到食物",
        "key_conditions": ["饥饿>50%", "食盆在环境中存在"],
    },
    "sleep": {
        "description": "睡觉/打盹 — 优先找猫窝，找不到则就地入睡15tick",
        "tree_type": "Selector",
        "main_path": "Sequence[精力<30%检查 → 安全环境检查 → Selector[找床Sequence/就地ForceSuccess] → 开始睡眠标记 → ProgressAction睡眠15tick → 醒来]",
        "fallback": "Sequence[环境不安全 → 保持警觉/报FAILURE]",
        "fail_behavior": "保持清醒，不睡",
        "key_conditions": ["精力<30%", "环境安全(恐惧<50%, 压力<60)"],
    },
    "play_with_toy": {
        "description": "玩玩具 — 寻路至最近玩具，玩耍4tick后短暂回味",
        "tree_type": "Selector",
        "main_path": "Sequence[玩具可用检查 → 找最近玩具 → Navigate移动 → ProgressAction玩耍4tick → ForceSuccess回味1-3tick]",
        "fallback": "Sequence[无玩具 → 报FAILURE]",
        "fail_behavior": "报告没有可玩的玩具",
        "key_conditions": ["玩具存在于环境中"],
    },
    "social_groom": {
        "description": "与某只猫互相舔毛 — 选择附近猫咪作为对象，持续6tick",
        "tree_type": "Selector",
        "main_path": "Sequence[附近有猫检查 → 选择社交对象 → 设目标为该猫位置 → Navigate移动 → ProgressAction舔毛6tick]",
        "fallback": "Sequence[无社交对象 → 报FAILURE]",
        "fail_behavior": "报告没有可以互相舔毛的伙伴",
        "key_conditions": ["附近有猫(≤3格)"],
    },
    "social_play": {
        "description": "与某只猫玩耍/追逐 — 选择玩伴进行追逐游戏，持续4tick",
        "tree_type": "Selector",
        "main_path": "Sequence[附近有猫检查 → 精力>50%检查 → 选择玩伴 → ProgressAction追逐4tick]",
        "fallback": "Sequence[无玩伴/无精力 → 报FAILURE]",
        "fail_behavior": "报告没有精力或玩伴",
        "key_conditions": ["附近有猫(≤3格)", "精力>50%"],
    },
    "hide": {
        "description": "躲藏 — 恐惧或压力高时寻隐蔽点蜷缩，无隐蔽点则就地蜷缩",
        "tree_type": "Selector",
        "main_path": "Sequence[恐惧>50%或压力>70检查 → 找最近隐蔽点 → Navigate移动 → 进入躲藏标记 → ProgressAction躲藏3-15tick]",
        "fallback": "Sequence[必须躲藏(恐惧>70%) → 就地蜷缩 → 等待 → 放松]",
        "fail_behavior": "无隐蔽点可用时蜷缩在角落",
        "key_conditions": ["恐惧>50%", "压力>70", "隐蔽点可用"],
    },
    "hiss_warning": {
        "description": "发出警告/哈气 — 感到威胁时发出警告，短暂对峙",
        "tree_type": "Sequence",
        "main_path": "Sequence[感到威胁检查(恐惧>60%或玩家逼近/抓取) → 哈气动作 → ProgressAction对峙1tick]",
        "fallback": "无",
        "fail_behavior": "停止哈气",
        "key_conditions": ["恐惧>60%", "玩家行为=approach/grab"],
    },
    "curious_inspect": {
        "description": "好奇探索某物/某处 — 随机选择物体或区域进行探索",
        "tree_type": "Selector",
        "main_path": "Sequence[好奇>60%检查 → 选择探索目标(加权随机) → Navigate移动 → ProgressAction探索3tick]",
        "fallback": "Sequence[无有趣目标 → 报FAILURE]",
        "fail_behavior": "报告没什么有趣的东西",
        "key_conditions": ["好奇>60%"],
    },
    "follow_player": {
        "description": "跟随玩家移动 — 高信任时尾随玩家，持续跟随",
        "tree_type": "Sequence",
        "main_path": "Sequence[信任≥60检查 → 玩家可见检查 → 设目标为玩家位置 → Navigate移动 → ProgressAction跟随2tick]",
        "fallback": "无",
        "fail_behavior": "停止跟随",
        "key_conditions": ["信任度≥60", "玩家在附近≤3格"],
    },
    "accept_petting": {
        "description": "接受抚摸/享受互动 — 玩家抚摸时接受并享受5tick",
        "tree_type": "Selector",
        "main_path": "Sequence[玩家动作=pet → 信任≥20检查 → 距离≤2检查 → 犹豫1tick → 接受抚摸 → ProgressAction享受5tick]",
        "fallback": "Sequence[拒绝抚摸 → 移开/报FAILURE]",
        "fail_behavior": "表示还不想被摸",
        "key_conditions": ["玩家动作=pet", "信任度≥20", "距离≤2"],
    },
    "fearful_retreat": {
        "description": "恐惧后退/躲避 — 计算远离玩家方向，快速后退并警惕观察",
        "tree_type": "Selector",
        "main_path": "Sequence[恐惧>40%检查 → 计算后退方向(远离玩家+性格flee距离) → 快速后退2步 → ProgressAction警惕观察2tick]",
        "fallback": "Sequence[无威胁感知 → 原地不动]",
        "fail_behavior": "原地不动",
        "key_conditions": ["恐惧>40%", "性格flee距离"],
    },
    "stare_at_window": {
        "description": "看窗外/发呆 — 找窗边位置发呆8tick，无窗户则原地发呆",
        "tree_type": "Selector",
        "main_path": "Sequence[平静检查(恐惧<40%, 压力<50) → 找最近窗边 → Navigate移动 → ProgressAction发呆8tick]",
        "fallback": "ForceSuccess[原地发呆4tick]",
        "fail_behavior": "原地发呆",
        "key_conditions": ["恐惧<40%", "压力<50", "窗边位置可用"],
    },
}
