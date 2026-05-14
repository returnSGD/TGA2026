# Luban 配置表操作指南

本文档是 `luban_helper.py` 脚本的完整命令参考，包含所有命令的参数详解、Excel 结构说明、数据填写格式和注释约定。

## 基础参数

**执行方式**：
```bash
python scripts/luban_helper.py <command> --data-dir Configs/GameConfig/Datas
```

**注意**：PowerShell 中使用分号 `;` 作为命令分隔符，不要使用 `&&`。

---

## 枚举操作

### enum list - 列出所有枚举
```bash
python scripts/luban_helper.py enum list --data-dir Configs/GameConfig/Datas
```

### enum get - 查询枚举详情
```bash
python scripts/luban_helper.py enum get test.ETestQuality --data-dir Configs/GameConfig/Datas
```

### enum add - 新增枚举
```bash
python scripts/luban_helper.py enum add test.EWeaponType --values "SWORD=1:剑,BOW=2:弓,STAFF=3:法杖" --comment "武器类型" --data-dir Configs/GameConfig/Datas
```

**参数**：
- `name`: 枚举全名（包含模块，如 `test.EWeaponType`）
- `--values`: 枚举值，格式 `name=value:alias,name2=value2:alias2`
- `--comment`: 枚举注释
- `--flags`: 是否为标志枚举（可选）

### enum delete - 删除枚举
```bash
python scripts/luban_helper.py enum delete test.EWeaponType --data-dir Configs/GameConfig/Datas
python scripts/luban_helper.py enum delete test.EWeaponType --force --data-dir Configs/GameConfig/Datas  # 强制删除
```

**参数**：
- `name`: 枚举名称
- `--force`: 强制删除，忽略引用检查

### enum update - 更新枚举属性
```bash
python scripts/luban_helper.py enum update test.EWeaponType --comment "武器类型枚举" --data-dir Configs/GameConfig/Datas
python scripts/luban_helper.py enum update test.EWeaponType --flags --data-dir Configs/GameConfig/Datas
```

---

## Bean 操作

### bean list - 列出所有 Bean
```bash
python scripts/luban_helper.py bean list --data-dir Configs/GameConfig/Datas
```

### bean get - 查询 Bean 详情
```bash
python scripts/luban_helper.py bean get test.TestBean1 --data-dir Configs/GameConfig/Datas
```

### bean add - 新增 Bean
```bash
python scripts/luban_helper.py bean add test.Weapon --fields "attack:int:攻击力,speed:float:攻击速度" --parent Item --comment "武器" --data-dir Configs/GameConfig/Datas
```

**参数**：
- `name`: Bean 全名（包含模块）
- `--fields`: 字段定义，格式 `name:type:comment,name2:type2:comment2`
- `--parent`: 父类名称（可选）
- `--comment`: Bean 注释（可选）

### bean delete - 删除 Bean
```bash
python scripts/luban_helper.py bean delete test.Weapon --data-dir Configs/GameConfig/Datas
python scripts/luban_helper.py bean delete test.Weapon --force --data-dir Configs/GameConfig/Datas  # 强制删除
```

### bean update - 更新 Bean 属性
```bash
python scripts/luban_helper.py bean update test.ItemList --sep "|" --data-dir Configs/GameConfig/Datas
python scripts/luban_helper.py bean update test.ItemList --comment "道具列表" --data-dir Configs/GameConfig/Datas
```

**参数**：
- `--sep`: 分隔符（用于 list 类型元素分隔）
- `--comment`: 注释
- `--alias`: 别名
- `--parent`: 父类名称

**分隔符说明**：

| 分隔符 | 数据格式示例 | 说明 |
|--------|-------------|------|
| 默认 `;` | `1001,10;2003,50;5003,10` | 默认分隔符 |
| `|` | `1001,10|2003,50|5003,10` | 更清晰（推荐） |
| `#` | `1001,10#2003,50#5003,10` | 自定义分隔符 |

---

## 表操作

### table list - 列出所有表
```bash
python scripts/luban_helper.py table list --data-dir Configs/GameConfig/Datas
```

### table get - 查询表详情
```bash
python scripts/luban_helper.py table get test.TbItem --data-dir Configs/GameConfig/Datas
```

### table add - 新增配置表
```bash
# 默认创建自动导入格式：#Item-道具表.xlsx
python scripts/luban_helper.py table add test.TbItem --fields "id:int:道具ID,name:string:道具名称" --comment "道具表" --data-dir Configs/GameConfig/Datas

# 使用传统方式（注册到 __tables__.xlsx）
python scripts/luban_helper.py table add test.TbItem --fields "id:int:道具ID,name:string:道具名称" --input "item.xlsx" --no-auto-import --data-dir Configs/GameConfig/Datas
```

**参数**：
- `name`: 表全名（包含模块，如 `test.TbItem`）
- `--fields`: 字段定义，格式 `name:type:comment:group`（group 可选）
- `--comment`: 表注释
- `--no-auto-import`: 使用传统方式
- `--vertical`: 使用纵表模式
- `--input`: 数据文件名
- `--sheet`: Sheet名称
- `--index`: 主键定义
- `--groups`: 分组列表

**自动导入格式**：
- 文件名格式：`#表名-注释.xlsx`
- Luban 自动识别，无需在 `__tables__.xlsx` 中声明

### table delete - 删除配置表
```bash
python scripts/luban_helper.py table delete test.TbItem --data-dir Configs/GameConfig/Datas
python scripts/luban_helper.py table delete test.TbItem --delete-data --data-dir Configs/GameConfig/Datas  # 同时删除数据文件
```

### table update - 更新表属性
```bash
python scripts/luban_helper.py table update test.TbItem --comment "道具配置表" --data-dir Configs/GameConfig/Datas
python scripts/luban_helper.py table update test.TbItem --input "item_v2.xlsx" --data-dir Configs/GameConfig/Datas
```

### 纵表（单例表）
```bash
python scripts/luban_helper.py table add test.TbGlobalConfig --fields "guild_open_level:int:公会开启等级,bag_init_size:int:初始格子数" --comment "全局配置" --vertical --data-dir Configs/GameConfig/Datas
```

**纵表结构**：
```
| ##column |          |          |         |
| ##var    | ##type   | ##       | ##group |
| guild_open_level | int | 公会开启等级 | c |
| bag_init_size    | int | 初始格子数   | c |
```

---

## 字段操作

### field list - 列出表的所有字段
```bash
python scripts/luban_helper.py field list test.TbItem --data-dir Configs/GameConfig/Datas
```

### field add - 添加字段
```bash
python scripts/luban_helper.py field add test.TbItem desc --type "string" --comment "道具描述" --data-dir Configs/GameConfig/Datas
```

**参数**：
- `table`: 表名称
- `name`: 字段名
- `--type`: 字段类型
- `--comment`: 字段注释（支持多行，用 `|` 分隔）
- `--group`: 字段分组（可选，不指定时自动推断）
- `--sheet`: Sheet名称
- `--position`: 插入位置（从0开始，-1表示末尾）

**分组自动推断规则**：
- `c` (客户端): name, desc, icon, image, model, effect, sound, ui 等显示相关
- `s` (服务器): server, logic, damage, hp, mp, exp, level, rate 等逻辑相关
- `cs` (两者): id, 其他无法明确判断的字段

### field update - 修改字段
```bash
python scripts/luban_helper.py field update test.TbItem desc --new-name "description" --comment "详细描述" --data-dir Configs/GameConfig/Datas
```

**参数**：
- `table`: 表名称
- `name`: 原字段名
- `--new-name`: 新字段名
- `--type`: 新类型
- `--comment`: 新注释
- `--group`: 新分组
- `--sheet`: Sheet名称

### field delete - 删除字段（危险操作）
```bash
python scripts/luban_helper.py field delete test.TbItem desc --data-dir Configs/GameConfig/Datas
```

**参数**：
- `--force`: 强制删除，跳过确认

**警告**：删除字段会同时删除该字段的所有数据。

### field disable / enable - 禁用/启用字段
```bash
python scripts/luban_helper.py field disable test.TbItem desc --data-dir Configs/GameConfig/Datas
python scripts/luban_helper.py field enable test.TbItem desc --data-dir Configs/GameConfig/Datas
```

禁用字段通过在字段名前添加 `##` 前缀实现，Luban 导表时会忽略该字段，但数据保留。

---

## 数据行操作

### row list - 列出数据行
```bash
python scripts/luban_helper.py row list test.TbItem --data-dir Configs/GameConfig/Datas
python scripts/luban_helper.py row list test.TbItem --start 10 --limit 20 --data-dir Configs/GameConfig/Datas
```

### row get - 按字段值查询数据行
```bash
python scripts/luban_helper.py row get TbItem --field id --value 1004 --data-dir Configs/GameConfig/Datas
python scripts/luban_helper.py row get TbItem --field name --value "屠龙刀" --data-dir Configs/GameConfig/Datas
```

**返回示例**：
```json
{"id": 1004, "name": "烈焰剑", "type": "Weapon", "quality": 4}
```

### row query - 多条件查询
```bash
python scripts/luban_helper.py row query TbItem --conditions '{"type":"Weapon","quality":5}' --data-dir Configs/GameConfig/Datas
python scripts/luban_helper.py row query TbItem --conditions '{"type":"Consumable"}' --limit 10 --data-dir Configs/GameConfig/Datas
```

### row add - 添加数据行
```bash
python scripts/luban_helper.py row add test.TbItem --data '{"id":1001,"name":"宝剑","count":1}' --data-dir Configs/GameConfig/Datas
```

**智能插入**：添加数据行时自动按 ID 顺序插入到合适位置，而非追加到末尾。
- ID 最大 → 追加到末尾
- ID 在中间 → 插入到合适位置

### row update - 更新数据行
```bash
python scripts/luban_helper.py row update test.TbItem 0 --data '{"name":"神剑"}' --data-dir Configs/GameConfig/Datas
```

### row delete - 删除数据行
```bash
python scripts/luban_helper.py row delete test.TbItem 0 --data-dir Configs/GameConfig/Datas
python scripts/luban_helper.py row delete test.TbItem 0 --force --data-dir Configs/GameConfig/Datas
```

---

## 批量操作

### batch fields - 批量添加字段
```bash
python scripts/luban_helper.py batch fields test.TbItem --data '[{"name":"price","type":"int","comment":"价格"},{"name":"quality","type":"int","comment":"品质"}]' --data-dir Configs/GameConfig/Datas
```

### batch rows - 批量添加数据行
```bash
python scripts/luban_helper.py batch rows test.TbItem --data '[{"id":1001,"name":"宝剑"},{"id":1002,"name":"铁剑"}]' --data-dir Configs/GameConfig/Datas
```

---

## 导入导出

### export - 导出表数据为 JSON
```bash
python scripts/luban_helper.py export test.TbItem --data-dir Configs/GameConfig/Datas
python scripts/luban_helper.py export test.TbItem --output item_backup.json --data-dir Configs/GameConfig/Datas
```

### import - 从 JSON 导入数据
```bash
python scripts/luban_helper.py import test.TbItem item_backup.json --data-dir Configs/GameConfig/Datas
```

**参数**：
- `--mode`: 导入模式（`append` 追加，`replace` 替换）

---

## 验证功能

### validate - 验证表数据
```bash
python scripts/luban_helper.py validate test.TbItem --data-dir Configs/GameConfig/Datas
python scripts/luban_helper.py validate --all --data-dir Configs/GameConfig/Datas
```

**验证内容**：
- 表结构完整性（##var、##type 行）
- 字段定义检查
- 数据类型验证

---

## 其他命令

### ref - 引用完整性检查
```bash
python scripts/luban_helper.py ref test.RewardItem --data-dir Configs/GameConfig/Datas
```

### template - 配置模板
```bash
python scripts/luban_helper.py template list --data-dir Configs/GameConfig/Datas
python scripts/luban_helper.py template create item TbEquip --module test --data-dir Configs/GameConfig/Datas
```

### rename - 重命名表
```bash
python scripts/luban_helper.py rename test.TbItem test.TbItemNew --data-dir Configs/GameConfig/Datas
python scripts/luban_helper.py rename test.TbItem test.TbItemNew --migrate-data --data-dir Configs/GameConfig/Datas
```

### copy - 复制表
```bash
python scripts/luban_helper.py copy test.TbItem test.TbItemCopy --data-dir Configs/GameConfig/Datas
```

### diff - 差异对比
```bash
python scripts/luban_helper.py diff test.TbItem test.TbItemV2 --data-dir Configs/GameConfig/Datas
```

### auto - 自动导入表操作
```bash
python scripts/luban_helper.py auto list --data-dir Configs/GameConfig/Datas
python scripts/luban_helper.py auto create #Item --fields "id:int:ID,name:string:名称" --data-dir Configs/GameConfig/Datas
```

### cache - 缓存操作
```bash
python scripts/luban_helper.py cache build --data-dir Configs/GameConfig/Datas
python scripts/luban_helper.py cache clear --data-dir Configs/GameConfig/Datas
```

---

## Excel 结构说明

### 数据表结构
```
| ##var   | id  | name   | count   |
| ##type  | int | string | int     |
| ##      | 道具ID | 道具名称 | 堆叠数量  |
| ##group | c   | c      | c       |  ← 可选
```

### 纵表结构（单例表）
```
| ##column |          |          |         |
| ##var    | ##type   | ##       | ##group |
| key      | string   | 配置键    | c       |
| value    | int      | 配置值    | s       |
```

### __enums__.xlsx 结构
- `full_name` 有值 = 枚举定义开始
- `full_name` 为空 = 上一枚举的枚举项
- `*items` 列（H列开始）= 枚举项数据

```
| ##var | full_name          | flags | unique | group | comment | tags | *items              |
|-------|-------------------|-------|--------|-------|---------|------|---------------------|
| ##var | name              | alias | value  | comment | tags  |      |                     |
| ##    | 全名              | 是否标志| 是否唯一 |       |         |      | 枚举名              |
|       | test.ETestQuality | False | True   |       |         |      | A | 白 | 1 | 最高品质 |
|       |                   |       |        |       |         |      | B | 黑 | 2 | 黑色的   |
```

### __beans__.xlsx 结构
- `full_name` 有值 = Bean 定义开始
- `full_name` 为空 = 上一 Bean 的字段
- `*fields` 列（J列开始）= 字段数据

```
| ##var | full_name          | parent | valueType | alias | sep | comment | tags | group | *fields      |
|-------|-------------------|--------|-----------|-------|-----|---------|------|-------|--------------|
| ##var | name              | alias  | type      | group | comment | tags | variants |            |
| ##    | 全名              | 父类   | 是否值类型 | 别名  | 分隔符 | 字段名  | 别名 | 类型  | 分组 | 注释 |
|       | test.TestBean1    |        |           |       |     | 测试Bean |      | c     | x1 | int | 最高品质 |
|       |                   |        |           |       |     |         |      |       | x2 | string | 黑色的 |
```

---

## 数据填写格式

### 基本类型
| 类型 | 格式示例 |
|------|---------|
| int | `100` |
| string | `道具名称` |
| bool | `true` 或 `false` |
| float | `1.5` |

### 枚举类型
```
Weapon      # 枚举名
1           # 数值
```

### List 类型
```
# list<int>
1;2;3;4;5

# list<RewardItem> 简写格式（推荐）
1001,100;1002,20
```

### Map 类型
```
# map<int,string>
1,金币;2,钻石;3,体力
```

---

## 注释约定规范

在 Luban Excel 表格中，`##` 开头的行是注释行，会被 Luban 忽略。约定：

**`##var` 行和数据起始行之间的所有 `##` 开头的行，作为字段注释。**

```
行号 | A列   | B列      | C列       | D列
-----|------|----------|----------|-----------
1    | ##var| id       | name     | count      ← 字段名行
2    | ##type| int     | string   | int        ← 类型行
3    | ##   | 道具ID   | 道具名称  | 堆叠数量    ← 注释行1
4    | ##   | 唯一标识 |          |            ← 注释行2（可多行）
5    | ##group| c      | c        | c          ← 分组行（可选）
6    |      | 1        | 道具1    | 10         ← 数据起始行
```

**优点**：
- **零改动**：完全兼容现有 Luban，`##` 本来就是注释
- **向后兼容**：现有表格无需修改
- **灵活**：支持多行注释
