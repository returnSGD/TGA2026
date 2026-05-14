# item.xlsx — 道具表数据

## Sheet: item

| ##var    | id    | name    | description | type        | quality | icon              | maxStack | sellPrice | reward        |
|----------|-------|---------|-------------|-------------|---------|-------------------|----------|-----------|---------------|
| ##type   | int   | string  | string      | EItemType   | EQuality| string            | int      | int       | Reward        |
| ##group  | c,s   | c,s     | c           | c,s         | c,s     | c                 | c,s      | c         | c,s           |
| ##comment| 道具ID| 名称    | 描述        | 物品类型    | 品质    | 图标路径          | 最大堆叠 | 出售价格  | 奖励          |
|          | 1001  | 生命药水| 恢复少量生命值| Consumable | White   | icons/potion_red  | 99       | 10        | 2001,1,1.0    |
|          | 1002  | 法力药水| 恢复少量法力值| Consumable | White   | icons/potion_blue | 99       | 10        | 2002,1,1.0    |
|          | 1003  | 铁剑    | 普通铁制长剑  | Weapon     | White   | icons/sword_iron  | 1        | 100       |               |
|          | 1004  | 钢甲    | 坚固钢制护甲  | Armor      | Green   | icons/armor_steel | 1        | 250       |               |
|          | 1005  | 皮革    | 普通皮革材料  | Material    | White   | icons/leather     | 999      | 5         |               |
|          | 1006  | 精华剑  | 精华打造的剑  | Weapon     | Blue    | icons/sword_elite | 1        | 500       | 2003,1,0.5    |
|          | 1007  | 龙鳞甲  | 传说龙鳞护甲  | Armor      | Purple  | icons/armor_dragon| 1        | 1000      | 2003,2,0.3    |

**字段说明**：
- `reward` 字段类型为 `Reward`（在 __beans__.xlsx 中定义的 Bean）
- Bean 数据在 Excel 中使用逗号分隔，按字段声明顺序对应
- `type` 和 `quality` 使用枚举类型（在 __enums__.xlsx 中定义）
- `c,s` 分组表示客户端和服务端共享该字段
