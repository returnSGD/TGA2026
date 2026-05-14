# TEngine 框架 Luban 集成指南

## 概述

TEngine 项目使用 Luban 作为配置表方案，生成格式为 **cs-bin**（C# 代码）+ **bin**（二进制数据），通过 YooAsset 资源系统加载二进制数据，生成代码位于热更程序集 `GameScripts/HotFix/GameProto/`。

## 项目目录结构

```
Configs/GameConfig/                           # 配置工程根目录
├── luban.conf                                # Luban 主配置文件
├── gen_code_bin_to_project.bat               # 客户端代码生成（懒加载模板）
├── gen_code_bin_to_project_lazyload.bat      # 客户端代码生成（标准模板）
├── gen_code_bin_to_server.bat                # 服务端代码生成
├── Defines/                                  # XML Schema 定义
│   └── builtin.xml                           # 内置类型（vector2/vector3 等）
├── Datas/                                    # Excel 数据源目录
│   ├── __tables__.xlsx                       # 表注册索引
│   ├── __beans__.xlsx                        # Bean 复合类型定义
│   ├── __enums__.xlsx                        # 枚举类型定义
│   └── item.xlsx                             # 业务数据表（示例）
└── CustomTemplate/                           # 自定义代码生成模板
    ├── ConfigSystem.cs                       # 配置加载器模板（非自动生成）
    ├── ExternalTypeUtil.cs                   # Unity 类型转换工具（非自动生成）
    └── CustomTemplate_Client_LazyLoad/       # 懒加载表模板
        └── cs-bin/tables.sbn

Tools/Luban/                                  # Luban 工具链
└── Luban.dll                                 # Luban 主程序

UnityProject/Assets/
├── GameScripts/HotFix/GameProto/             # 热更程序集（生成代码）
│   ├── ConfigSystem.cs                       # 配置加载器
│   ├── ExternalTypeUtil.cs                   # Unity 类型映射
│   └── GameConfig/                           # Luban 自动生成的代码
│       ├── Tables.cs                         # 表管理类
│       └── item/                             # 按模块组织的生成代码
│           ├── TbItem.cs                     # 表类
│           ├── Item.cs                       # 数据类
│           ├── ItemExchange.cs               # 关联数据类
│           └── EQuality.cs                   # 枚举类
└── AssetRaw/Configs/bytes/                   # 二进制配置数据（YooAsset 管理）
    └── item_tbitem.bytes                     # Item 表二进制数据
```

## luban.conf 实际配置

项目使用 Excel 定义方式（`__tables__.xlsx`/`__beans__.xlsx`/`__enums__.xlsx`），搭配 XML Schema（`Defines/`）：

```json
{
    "groups": [
        {"names": ["c"], "default": true},
        {"names": ["s"], "default": true},
        {"names": ["e"], "default": true}
    ],
    "schemaFiles": [
        {"fileName": "Defines", "type": ""},
        {"fileName": "Datas/__tables__.xlsx", "type": "table"},
        {"fileName": "Datas/__beans__.xlsx", "type": "bean"},
        {"fileName": "Datas/__enums__.xlsx", "type": "enum"}
    ],
    "dataDir": "Datas",
    "targets": [
        {"name": "server", "manager": "Tables", "groups": ["s"], "topModule": "GameConfig"},
        {"name": "client", "manager": "Tables", "groups": ["c"], "topModule": "GameConfig"},
        {"name": "all", "manager": "Tables", "groups": ["c,s,e"], "topModule": "GameConfig"}
    ]
}
```

**关键约定**：
- `topModule` 为 `GameConfig`，生成代码命名空间为 `GameConfig.xxx`
- 分组 `c`（客户端）、`s`（服务端）、`e`（编辑器）
- 代码输出目标：`cs-bin`，数据输出目标：`bin`

## 代码生成脚本（导出数据的唯一入口）

**导出数据时，始终使用项目提供的脚本，不要手动拼接 dotnet 命令。**

脚本位于 `Configs/GameConfig/` 目录下，使用相对路径调用：

| 脚本 | 用途 | 说明 |
|:---|:---|:---|
| `gen_code_bin_to_project_lazyload` | 客户端代码+数据（懒加载模板，**推荐**） | AI 调用此脚本 |
| `gen_code_bin_to_project` | 客户端代码+数据（标准模板） | 非懒加载 |
| `gen_code_bin_to_server` | 服务端代码+数据 | - |

### AI 调用导表命令

根据操作系统选择对应扩展名：

**Windows：**
```bash
cmd //c "set AI_MODE=1 && Configs/GameConfig/gen_code_bin_to_project_lazyload.bat"
```

**macOS/Linux：**
```bash
bash Configs/GameConfig/gen_code_bin_to_project_lazyload.sh
```

### 客户端生成脚本

**位置**：`Configs/GameConfig/gen_code_bin_to_project_lazyload.bat`（Windows）/ `.sh`（macOS/Linux）

**脚本流程**：
1. 复制 `CustomTemplate/ConfigSystem.cs` → `GameProto/ConfigSystem.cs`
2. 复制 `CustomTemplate/ExternalTypeUtil.cs` → `GameProto/ExternalTypeUtil.cs`
3. 执行 Luban 代码生成（`--customTemplateDir` 启用懒加载模板）

### 服务端生成脚本

**位置**：`Configs/GameConfig/gen_code_bin_to_server.bat`（Windows）/ `.sh`（macOS/Linux）

**注意事项**：
- 懒加载脚本使用 `--customTemplateDir` 启用懒加载模板（表数据首次访问时才加载）
- `xcopy`/`cp` 复制的 `ConfigSystem.cs` 和 `ExternalTypeUtil.cs` 不是 Luban 自动生成的
- 生成前确保 `.NET SDK 8.0+` 已安装

## ConfigSystem 配置加载器

`ConfigSystem.cs` 桥接 Luban 生成代码与 TEngine 的 YooAsset 资源系统：

```csharp
using Luban;
using GameConfig;
using TEngine;
using UnityEngine;

public class ConfigSystem
{
    private static ConfigSystem _instance;
    public static ConfigSystem Instance => _instance ??= new ConfigSystem();

    private bool _init = false;
    private Tables _tables;

    /// <summary>懒加载访问所有配置表</summary>
    public Tables Tables
    {
        get
        {
            if (!_init) Load();
            return _tables;
        }
    }

    private IResourceModule _resourceModule;

    public void Load()
    {
        _tables = new Tables(LoadByteBuf);
        _init = true;
    }

    private ByteBuf LoadByteBuf(string file)
    {
        if (_resourceModule == null)
        {
            _resourceModule = ModuleSystem.GetModule<IResourceModule>();
        }
        TextAsset textAsset = _resourceModule.LoadAsset<TextAsset>(file);
        return new ByteBuf(textAsset.bytes);
    }
}
```

### 初始化时机

配置表在 `ProcedurePreload` 阶段由 YooAsset 预加载 `.bytes` 资源。首次访问 `ConfigSystem.Instance.Tables` 时自动触发加载。

## ExternalTypeUtil Unity 类型映射

通过 `Defines/builtin.xml` 的 `mapper` 定义，将 Luban 配置类型映射到 Unity 内置类型：

```csharp
public static class ExternalTypeUtil
{
    public static Vector2 NewVector2(GameConfig.vector2 v)
        => new Vector2(v.X, v.Y);

    public static Vector3 NewVector3(GameConfig.vector3 v)
        => new Vector3(v.X, v.Y, v.Z);

    public static Vector4 NewVector4(GameConfig.vector4 v)
        => new Vector4(v.X, v.Y, v.Z, v.W);

    public static Vector2Int NewVector2Int(GameConfig.vector2int v)
        => new Vector2Int(v.X, v.Y);

    public static Vector3Int NewVector3Int(GameConfig.vector3int v)
        => new Vector3Int(v.X, v.Y, v.Z);
}
```

对应的 XML 定义（`Defines/builtin.xml`）：

```xml
<module name="">
    <bean name="vector3" valueType="1" sep=",">
        <var name="x" type="float"/>
        <var name="y" type="float"/>
        <var name="z" type="float"/>
        <mapper target="client" codeTarget="cs-bin,cs-simple-json,cs-newtonsoft-json">
            <option name="type" value="UnityEngine.Vector3"/>
            <option name="constructor" value="ExternalTypeUtil.NewVector3"/>
        </mapper>
    </bean>
</module>
```

## 配置数据访问

### 基础访问

```csharp
// 获取表实例
var tables = ConfigSystem.Instance.Tables;

// Map 表：按主键查询
var itemCfg = tables.TbItem.Get(1001);

// 遍历所有数据
foreach (var item in tables.TbItem.DataList)
{
    Log.Info($"{item.Id} - {item.Name}");
}
```

### 推荐封装 ConfigManager

复杂模块应封装配置管理器，避免业务代码直接散落 `ConfigSystem.Instance.Tables.TbXxx`：

```csharp
// GameLogic/Config/ItemConfigMgr.cs
public class ItemConfigMgr
{
    private static ItemConfigMgr _instance;
    public static ItemConfigMgr Instance => _instance ??= new ItemConfigMgr();

    /// <summary>获取物品配置</summary>
    public Item GetItem(int id)
    {
        return ConfigSystem.Instance.Tables.TbItem.Get(id);
    }

    /// <summary>按品质筛选物品</summary>
    public List<Item> GetItemsByQuality(EQuality quality)
    {
        return ConfigSystem.Instance.Tables.TbItem.DataList
            .Where(i => i.Quality == quality)
            .ToList();
    }
}
```

**使用**：
```csharp
// 推荐
var item = ItemConfigMgr.Instance.GetItem(1001);

// 不推荐（简单场景可直接用）
var item = ConfigSystem.Instance.Tables.TbItem.Get(1001);
```

## Excel 表定义规范（TEngine 方式）

### __tables__.xlsx 注册表

| full_name | value_type | read_mode | comment |
|-----------|-----------|-----------|---------|
| cfg.TbItem | Item | map | 道具表 |
| cfg.TbSkill | Skill | map | 技能表 |
| cfg.TbLevel | Level | list | 关卡表 |

- `full_name`：`cfg.表名`，生成 `GameConfig.Tables.TbXxx`
- `value_type`：对应的数据类名
- `read_mode`：`map`（按键索引）/ `list`（列表）

### __beans__.xlsx 复合类型

| full_name | fields.name | fields.type | fields.comment |
|-----------|-------------|-------------|----------------|
| ItemDrop | itemId | int | 道具ID |
| ItemDrop | count | int | 数量 |
| ItemDrop | probability | float | 概率 |

在数据表中使用 `ItemDrop` 或 `ItemDrop[]`。

### __enums__.xlsx 枚举

| enum_name | item_name | item_value | item_alias |
|-----------|-----------|------------|------------|
| EQuality | White | 0 | 白 |
| EQuality | Green | 1 | 绿 |

### 业务数据表格式

```
第1行（字段名）: id    | name   | hp   | atk  | desc
第2行（类型）:   int   | string | int  | int  | string
第3行（分组）:   c     | c      | c    | c    | c
第4行（注释）:   道具ID| 名称   | 血量 | 攻击 | 描述
第5行+（数据）: 1001  | 木剑   | 0    | 10   | 新手木剑
```

## 添加新配置表完整流程

```
1. 在 Datas/__tables__.xlsx 中注册新表
   full_name: cfg.TbNewTable   value_type: NewTableRow   read_mode: map

2. 创建 Datas/new_table.xlsx，按规范添加列定义和数据

3. （如需复合类型）在 __beans__.xlsx 中定义 Bean
   （如需枚举）在 __enums__.xlsx 中定义枚举
   （如需 Unity 类型）在 Defines/builtin.xml 中添加 mapper

4. 运行 gen_code_bin_to_project.bat 生成代码和数据

5. 验证生成结果：
   - GameProto/GameConfig/ 下新增 TbNewTable.cs 和 NewTableRow.cs
   - AssetRaw/Configs/bytes/ 下新增 new_table_tbnewtable.bytes

6. 在 Unity Editor 中确认 .bytes 文件被 YooAsset 正确收集

7. 在 GameLogic/Config/ 下创建 NewTableConfigMgr.cs 封装查询方法
```

## 兼容性注意事项

- **新增字段**：前向兼容，旧客户端自动忽略新字段
- **删除字段**：不兼容，旧客户端会报错
- **修改字段类型**：不兼容，需同步更新代码
- **重命名字段**：不兼容，影响热更包
- 生成代码（`GameConfig/` 目录）**不要手动修改**，下次生成会覆盖
- `ConfigSystem.cs` 和 `ExternalTypeUtil.cs` 是手动维护的桥接文件，位于 `CustomTemplate/` 模板中
