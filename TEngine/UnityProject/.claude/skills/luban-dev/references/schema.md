# Luban Schema 定义详解

## XML Schema 定义

### 文件基本结构

```xml
<?xml version="1.0" encoding="utf-8"?>
<module name="模块名">

    <!-- 模块导入（可选） -->
    <import name="other_module"/>

    <!-- 枚举定义 -->
    <enum name="EItemType" comment="物品类型">
        <var name="Weapon" alias="武器" value="0"/>
        <var name="Armor" alias="护甲" value="1"/>
    </enum>

    <!-- 结构体定义 -->
    <bean name="ItemBase" comment="物品基础">
        <var name="id" type="string" comment="ID"/>
        <var name="name" type="string" comment="名称"/>
    </bean>

    <!-- 表定义 -->
    <table name="TbItem" value="ItemBase"
           input="item/Item.csv" mode="map" index="id"/>

</module>
```

## 枚举定义 (enum)

### 基本语法

```xml
<enum name="EQuality" comment="品质" flags="false">
    <var name="White" alias="白" value="0"/>
    <var name="Green" alias="绿" value="1"/>
    <var name="Blue" alias="蓝" value="2"/>
</enum>

<!-- Flags 枚举（位运算） -->
<enum name="EEquipSlot" flags="true">
    <var name="Head" value="1"/>      <!-- 0b0001 -->
    <var name="Body" value="2"/>      <!-- 0b0010 -->
    <var name="Hand" value="4"/>      <!-- 0b0100 -->
</enum>
```

### 属性说明

| 属性 | 说明 |
|------|------|
| `name` | 枚举名 |
| `flags` | 是否为标志位枚举（对应 C# FlagsAttribute）|
| `unique` | 值是否唯一（用于 ID 类枚举）|
| `comment` | 注释 |
| `alias` | 别名（用于 Excel 显示）|

### 枚举值自动递增

```xml
<enum name="AutoIncrement">
    <var name="A" value="1"/>   <!-- 显式指定 1 -->
    <var name="B"/>             <!-- 自动 2 -->
    <var name="C"/>             <!-- 自动 3 -->
</enum>
```

## 结构体定义 (bean)

### 基础 bean

```xml
<bean name="ItemCfg" comment="物品配置">
    <var name="id" type="string" comment="ID"/>
    <var name="name" type="string" comment="名称"/>
    <var name="price" type="int#range=[0,)" comment="价格"/>
</bean>
```

### 继承（父 bean 自动为抽象类）

```xml
<bean name="WeaponCfg" parent="ItemCfg" comment="武器配置">
    <var name="damage" type="int" comment="伤害"/>
    <var name="attackSpeed" type="float" comment="攻速"/>
</bean>
```

### 多态 bean

```xml
<bean name="EffectConfig" comment="效果配置" sep=",">
    <var name="type" type="string" comment="效果类型"/>
</bean>

<bean name="DamageEffect" parent="EffectConfig">
    <var name="damage" type="int"/>
    <var name="damageType" type="string"/>
</bean>

<bean name="HealEffect" parent="EffectConfig">
    <var name="healAmount" type="int"/>
</bean>
```

### bean 属性说明

| 属性 | 说明 |
|------|------|
| `name` | bean 名 |
| `parent` | 父 bean 名（支持继承和多态）|
| `valueType` | 是否为值类型（C# struct）|
| `sep` | 分隔符，用于紧凑格式数据 |
| `group` | 分组控制 |

## 字段定义 (var)

### 基本字段

```xml
<var name="id" type="int" comment="ID"/>
<var name="name" type="string" comment="名称"/>
<var name="quality" type="Quality" comment="品质"/>
```

### 带校验器的字段

```xml
<var name="level" type="int#range=[1,100]" comment="等级"/>
<var name="itemId" type="int#ref=item.TbItem" comment="物品ID"/>
<var name="count" type="int#!" comment="数量（不能为0）"/>
```

### 字段属性

| 属性 | 说明 |
|------|------|
| `name` | 字段名 |
| `type` | 字段类型 |
| `comment` | 注释 |
| `groups` | 导出分组 |
| `variants` | 变体列表 |

## 表定义 (table)

### Map 表

```xml
<table name="TbItem" value="ItemCfg"
       input="item/Item.csv" mode="map" index="id"/>
```

### List 表

```xml
<table name="TbDropTable" value="DropItem"
       input="drop/*.csv" mode="list"/>
```

### 单例表

```xml
<table name="TbGlobalConfig" value="GlobalConfig"
       input="GlobalConfig.json" mode="one"/>
```

### 表属性说明

| 属性 | 必填 | 说明 |
|------|------|------|
| `name` | 是 | 生成的表类名，推荐 `TbXxx` 格式 |
| `value` | 是 | value_type，即 bean 名 |
| `input` | 是 | 数据文件路径，支持通配符 `*` |
| `mode` | 否 | 容器类型：`map`/`list`/`one`，默认 `map` |
| `index` | 否 | map 模式的主键字段名 |
| `output` | 否 | 输出文件名 |
| `readSchemaFromFile` | 否 | 从数据文件读取 schema |

### Mode 说明

| 模式 | 说明 | 数据格式 |
|------|------|----------|
| `map` | 字典，按 key 索引 | `{ "key1": {...}, "key2": {...} }` |
| `list` | 列表 | `[ {...}, {...} ]` |
| `one` | 单条记录 | `{ ... }` |

### 索引类型

```xml
<!-- 单字段索引 -->
<table index="id" mode="map"/>

<!-- 联合索引（组合键） -->
<table index="key1+key2+key3" mode="list"/>

<!-- 独立索引（多个索引） -->
<table index="key1,key2,key3" mode="list"/>
```

## 类型引用

### 同模块引用

```xml
<var name="type" type="EItemType"/>
<var name="weapon" type="WeaponCfg"/>
```

### 跨模块引用

```xml
<var name="monsterId" type="string#ref=monster.TbMonster"/>
<var name="weapon" type="item.WeaponCfg"/>
```

## refgroup 定义

```xml
<refgroup name="EquipTables">
    item.TbWeapon,item.TbArmor,item.TbAccessory
</refgroup>

<var name="equipId" type="string#ref=EquipTables"/>
```

## Excel Schema 定义

### __enums__.xlsx

```
enum_name | isFlags | comment
Quality   | false   | 品质

enum_name | item_name | item_value | item_alias
Quality   | WHITE     | 0          | 白
Quality   | GREEN     | 1          | 绿
```

### __beans__.xlsx

```
bean_name | parent | comment
Item      |        | 物品

bean_name | field_name | field_type | field_comment
Item      | id         | int        | ID
Item      | name       | string     | 名称
```

### __tables__.xlsx

```
table_name | value_type | input          | mode | index
TbItem     | Item       | Item.xlsx      | map  | id
TbQuest    | Quest      | Quest.xlsx     | list |
```
