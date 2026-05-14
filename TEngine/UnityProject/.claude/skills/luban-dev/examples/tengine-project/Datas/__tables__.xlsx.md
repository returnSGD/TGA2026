# __tables__.xlsx — 表注册索引

本文件注册所有业务数据表。Luban 根据此文件生成对应的 TbXxx 表类。

## Sheet: table

| ##var   | name       | value_type | input      | mode | index | group | comment |
|---------|------------|------------|------------|------|-------|-------|---------|
| ##type  | string     | string     | string     | string| string| string| string  |
|         | cfg.TbItem | Item       | item.xlsx  | map  | id    |       | 道具表  |
|         | cfg.TbSkill| Skill      | skill.xlsx | map  | id    |       | 技能表  |

**字段说明**：
- `name`：`cfg.TbXxx` 格式，生成 `GameConfig.Tables.TbXxx`
- `value_type`：数据行对应的类型名（需在数据表第一行定义或引用 Bean）
- `input`：数据文件名（相对于 Datas/ 目录）
- `mode`：`map`（按主键索引）/ `list`（列表）
- `index`：map 模式的主键字段名
