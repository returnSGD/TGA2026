# skill.xlsx — 技能表数据

## Sheet: skill

| ##var    | id       | name    | description      | skillType  | damageType | cooldown | manaCost | castTime | range | drops              |
|----------|----------|---------|------------------|------------|------------|----------|----------|----------|-------|---------------------|
| ##type   | int      | string  | string           | ESkillType | EDamageType| float    | int      | float    | float | DropItem[]          |
| ##group  | c,s      | c,s     | c                | c,s        | c          | c,s      | c,s      | c        | c     | c,s                 |
| ##comment| 技能ID   | 名称    | 描述             | 技能类型    | 伤害类型    | 冷却时间 | 法力消耗 | 施法时间 | 距离  | 掉落列表            |
|          | 5001     | 火球术  | 发射火球造成伤害  | Active     | Fire       | 3.0      | 20       | 0.5      | 10.0  |                     |
|          | 5002     | 冰箭    | 发射冰箭冻结目标  | Active     | Ice        | 5.0      | 30       | 0.8      | 12.0  |                     |
|          | 5003     | 雷击    | 召唤闪电攻击敌人  | Active     | Lightning  | 8.0      | 50       | 1.0      | 15.0  |                     |
|          | 6001     | 铁壁    | 提升防御力       | Passive    | Physical   | 0        | 0        | 0        | 0     |                     |
|          | 7001     | 激励    | 提升攻击力       | Buff       | Physical   | 0        | 0        | 0        | 0     | 1005,1,5;1005,2,10  |

**字段说明**：
- `drops` 类型为 `DropItem[]`（Bean 数组，多个 DropItem 用分号分隔，单个内部用逗号分隔）
- `DropItem` 定义：`itemId,minCount,maxCount,weight`
- 示例 `1005,1,5;1005,2,10` 表示两个掉落项：
  - 物品1005，数量1~5，权重10
  - 物品1005，数量2~10，权重10
