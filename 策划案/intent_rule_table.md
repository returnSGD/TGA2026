# 猫咪意图规则表（策划填写版）

## 一、表结构说明

每条规则 = 一组条件 → 一个意图。条件维度如下：

| 维度 | 字段 | 说明 |
|------|------|------|
| 性格 | clingy(黏人), timid(怯懦), greedy(贪吃), playful(活泼), social(社交), aggressive(攻击性), active(活跃), independent(独立) | 0.0~1.0，猫咪终身不变 |
| 情绪 | hunger(饥饿), fear(恐惧), curiosity(好奇), comfort(舒适), social(社交需求) | 0.0~1.0，实时波动 |
| 信任度 | trust | 0~100，对玩家的累计信任 |
| 环境 | comfort(区域舒适度), stimulation(刺激度), noise(噪音), light(光线), hygiene(卫生) | 0.0~1.0 |
| 玩家行为 | player_action | 枚举：pet/feed/call/play/ignore/scold/approach/leave/treat/heal |
| 玩家距离 | player_distance | near(近距离)/mid(中距离)/far(远距离) |
| 陌生人 | stranger_present | true/false，当前场景是否有陌生人 |
| → 意图 | intent | 输出的意图枚举 |

---

## 二、策划填写样例（小雪：高怯懦流浪猫）

猫咪：**小雪** | 性格锚点：`怯懦=0.9, 独立=0.8, 黏人=0.1, 好奇=0.1, 社交=0.1`

### 场景A：面对主人（信任度 > 40）

| # | 情绪条件 | 信任度 | 环境 | 玩家行为 | 玩家距离 | 陌生人 | → 意图 | 备注 |
|---|---------|--------|------|---------|---------|--------|--------|------|
| A1 | fear>0.6 | 40~60 | — | approach | near | false | **fearful_retreat** | 信任不够，本能躲 |
| A2 | fear>0.6 | 40~60 | — | approach | far | false | **fearful_hesitate** | 远距离观察，犹豫不决 |
| A3 | fear>0.6 | 40~60 | — | crouch(安静蹲下) | far | false | **curious_inspect** | 降低姿态会触发好奇试探 |
| A4 | fear 0.3~0.6 | 50~70 | comfort>0.6 | pet | near | false | **freeze_tolerate** | 不敢逃但僵住接受抚摸 |
| A5 | fear<0.3 | >70 | comfort>0.6 | pet | near | false | **accept_petting** | 终于安心接受抚摸 |
| A6 | fear<0.3 | >70 | — | treat(feed) | near | false | **cautious_approach** | 想吃但还有戒心，缓慢靠近 |
| A7 | hunger>0.5 | >50 | — | feed(放下食物离开) | far | false | **wait_then_eat** | 等人走远才去吃 |
| A8 | comfort>0.5 | >60 | light<0.3 | ignore | far | false | **sleep** | 昏暗环境独自入睡 |
| A9 | curiosity>0.4 | >50 | stimulation>0.4 | play(挥逗猫棒) | mid | false | **paw_reach** | 伸爪试探但不离开安全位置 |

### 场景B：面对陌生人或低信任状态

| # | 情绪条件 | 信任度 | 环境 | 玩家行为 | 玩家距离 | 陌生人 | → 意图 | 备注 |
|---|---------|--------|------|---------|---------|--------|--------|------|
| B1 | fear>0.5 | <30 | — | approach | any | true | **hide** | 陌生人出现，立刻躲藏 |
| B2 | fear>0.5 | <30 | — | approach | near | false | **fearful_retreat** | 不熟的人靠近 |
| B3 | fear>0.7 | <20 | — | scold | any | any | **flee** | 被呵斥，直接逃跑 |
| B4 | fear>0.8 | <20 | — | call | any | true | **hide** | 极度恐惧时叫也不出来 |
| B5 | fear 0.3~0.6 | 30~50 | — | approach | near | false | **fearful_hesitate** | 想逃但还在犹豫 |
| B6 | hunger>0.7 | <30 | — | feed(放下食物) | far | false | **wait_then_eat** | 饿极了也会等安全才吃 |
| B7 | fear>0.5 | <40 | noise>0.5 | ignore | far | true | **hide** | 嘈杂环境+陌生人，躲 |

### 场景C：夜间/安静时刻（小雪放松时）

| # | 情绪条件 | 信任度 | 环境 | 玩家行为 | 玩家距离 | 陌生人 | → 意图 | 备注 |
|---|---------|--------|------|---------|---------|--------|--------|------|
| C1 | fear<0.3, comfort>0.7 | >70 | light<0.2 | ignore | far | false | **sleep** | 夜深人静安心睡 |
| C2 | fear<0.2, comfort>0.6 | >70 | light<0.2 | sit_quietly | far | false | **slow_blink** | 隔空对玩家眯眼（信任的最高信号） |
| C3 | comfort>0.8 | >80 | — | pet | near | false | **purr** | 呼噜回应 |
| C4 | curiosity>0.3, fear<0.3 | >60 | stimulation>0.3 | play | mid | false | **investigate_toy** | 主动靠近玩具 |

---

## 三、策划填写样例（橘墩：高黏人、低胆怯）

猫咪：**橘墩** | 性格锚点：`黏人=0.9, 贪吃=0.95, 活跃=0.7, 怯懦=0.1, 社交=0.8`

| # | 情绪条件 | 信任度 | 环境 | 玩家行为 | 玩家距离 | 陌生人 | → 意图 | 备注 |
|---|---------|--------|------|---------|---------|--------|--------|------|
| D1 | hunger>0.3 | >20 | — | enter_room | any | false | **demand_food** | 一进门就叫着要吃的 |
| D2 | social>0.3 | >30 | — | sit_quietly | mid | false | **friendly_approach** | 主动过来蹭 |
| D3 | comfort>0.4 | >30 | — | pet | near | false | **purr + rub_leg** | 抚摸就呼噜蹭腿 |
| D4 | hunger>0.7 | >10 | — | feed | near | false | **rush_to_eat** | 不管不顾冲去吃 |
| D5 | — | >20 | — | play | near | false | **playful_pounce** | 扑向玩具 |
| D6 | social>0.5 | >40 | stranger=true | stranger_approach | near | true | **curious_sniff** | 对陌生人也不怕，好奇过去闻 |
| D7 | hunger>0.3 | >30 | — | ignore | near | false | **persistent_meow** | 被无视也不放弃，持续叫 |
| D8 | social>0.4 | >50 | — | approach(其他猫) | — | — | **social_groom** | 主动给其他猫舔毛 |

### 橘墩 vs 小雪——同一场景，不同性格，不同意图（关键对比）

| 场景 | 小雪（怯懦=0.9） | 橘墩（怯懦=0.1, 黏人=0.9） |
|------|-----------------|---------------------------|
| 玩家推门进来 | **hide** 躲到柜子后面 | **friendly_approach** 小跑过来蹭腿 |
| 玩家蹲下伸手 | **fearful_hesitate** 犹豫观察 | **accept_petting** 把头塞进手心 |
| 陌生客人进门 | **flee** 瞬间消失 | **curious_sniff** 凑过去闻 |
| 玩家放下食物 | **wait_then_eat** 等没人了才吃 | **rush_to_eat** 直接冲过去吃 |
| 被玩家呵斥 | **flee** 逃跑，很久不出来 | **confused_stare** 歪头看你，不明白 |
| 夜深熄灯后 | **sleep** 独自缩在角落睡 | **cuddle_next_to_player** 悄悄跳上床蹭着睡 |

---

## 四、意图枚举（完整30个）

```
# 日常类
idle_wander       # 随意走动
sleep             # 睡觉
groom_self        # 自己舔毛
stare_at_window   # 望窗外
scratch_furniture # 磨爪

# 亲近类
friendly_approach # 主动靠近
rub_leg           # 蹭腿
accept_petting    # 接受抚摸
purr              # 呼噜
slow_blink        # 眯眼（信任信号）
cuddle_next_to_player # 贴着玩家睡
follow_player     # 跟随玩家

# 恐惧/防御类
fearful_retreat   # 恐惧后退
fearful_hesitate  # 恐惧犹豫（想逃但还在观察）
freeze_tolerate   # 僵住忍受（不敢动）
hide              # 躲藏
flee              # 逃跑
hiss_warning      # 哈气警告
tail_lash         # 甩尾（烦躁信号）

# 饥饿/食物类
demand_food       # 主动讨食
rush_to_eat       # 冲去吃
wait_then_eat     # 等待安全后再吃
cautious_approach # 谨慎靠近食物

# 玩耍/好奇类
playful_pounce    # 扑玩具
investigate_toy   # 试探玩具
investigate_sound # 探察声音
curious_inspect   # 好奇观察
curious_sniff     # 凑近闻
paw_reach         # 伸爪试探

# 社交类（猫猫之间）
social_groom      # 互相舔毛
play_wrestle      # 打闹玩耍
```

---

## 五、规则填写指引

策划按这个优先级顺序填表即可覆盖核心场景：

1. **第一步**：为每只初始猫咪填「性格对比表」（第四节那种对比，横向对比确保差异化）
2. **第二步**：按「信任度阶段」填——低信任/中信任/高信任，三段的同一场景要有递进变化
3. **第三步**：按「玩家行为」填——覆盖高频行为（pet/feed/approach/ignore/play）即可，低频行为可后补
4. **第四步**：按「特殊环境」填——夜间/嘈杂/有陌生人/有其他猫在场

**预估工作量**：每只猫约 40-60 条规则（覆盖核心场景），3 只初始猫共 120-180 条策划规则 → 脚本扩增至 5000+ 条训练样本。
