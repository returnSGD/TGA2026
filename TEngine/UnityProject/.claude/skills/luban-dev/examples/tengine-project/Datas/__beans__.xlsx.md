# __beans__.xlsx — Bean 复合类型定义

本文件定义复合数据结构（类似 C# struct/class）。

## Sheet: bean

| ##var        | name    | fields.name | fields.type | fields.comment |
|-------------|---------|-------------|-------------|----------------|
| ##type      | string  | string      | string      | string         |
|             | Reward  | itemId      | int         | 道具ID         |
|             | Reward  | count       | int         | 数量           |
|             | Reward  | probability | float       | 概率           |
|             | DropItem| itemId      | int         | 掉落物品ID     |
|             | DropItem| minCount    | int         | 最小数量       |
|             | DropItem| maxCount    | int         | 最大数量       |
|             | DropItem| weight      | int         | 掉落权重       |

**使用方式**：
- 在数据表中直接使用 `Reward` 类型
- 使用 `Reward[]` 表示 Reward 数组（Excel 中用逗号分隔或使用多行格式）

**Excel 数据格式示例**（在 item.xlsx 中使用 Reward）：
```
| id   | name   | reward                |
| int  | string | Reward                |
| 1001 | 宝箱   | 2001,5,0.8            |
```
这里 `2001,5,0.8` 依次对应 Reward 的三个字段 itemId,count,probability。
