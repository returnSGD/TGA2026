# TEngine 项目 Luban 配置示例

本示例演示了符合 TEngine 框架实际项目结构的 Luban 配置表开发方式。

## 目录结构

```
tengine-project/
├── luban.conf                          # Luban 主配置
├── gen_code_bin_to_project.bat         # 客户端导出脚本（懒加载模板，推荐）
├── gen_code_bin_to_server.bat          # 服务端导出脚本
├── Defines/                            # XML Schema（内置类型 + 自定义类型）
│   └── builtin.xml                     # Unity 类型映射（vector2/vector3 等）
├── Datas/                              # Excel 数据源
│   ├── __tables__.xlsx                 # 表注册索引（表格描述）
│   ├── __beans__.xlsx                  # Bean 复合类型定义
│   ├── __enums__.xlsx                  # 枚举类型定义
│   ├── item.xlsx                       # 道具表数据
│   └── skill.xlsx                      # 技能表数据
├── CustomTemplate/                     # 自定义模板
│   ├── ConfigSystem.cs                 # 配置加载器桥接文件
│   └── ExternalTypeUtil.cs             # Unity 类型转换
└── GameLogic/Config/                   # 配置管理器（业务代码）
    └── ItemConfigMgr.cs                # 道具配置管理器示例
```

## 与项目实际路径的对应关系

| 示例路径 | 项目实际路径 |
|---------|-------------|
| `luban.conf` | `Configs/GameConfig/luban.conf` |
| `gen_code_bin_to_project.bat` | `Configs/GameConfig/gen_code_bin_to_project.bat` |
| `gen_code_bin_to_server.bat` | `Configs/GameConfig/gen_code_bin_to_server.bat` |
| `Defines/builtin.xml` | `Configs/GameConfig/Defines/builtin.xml` |
| `Datas/*.xlsx` | `Configs/GameConfig/Datas/*.xlsx` |
| `CustomTemplate/` | `Configs/GameConfig/CustomTemplate/` |
| `GameLogic/Config/` | `UnityProject/Assets/GameScripts/HotFix/GameLogic/Config/` |

## 数据导出（使用导出脚本）

**不要手动拼接 dotnet 命令，直接运行 bat 脚本即可：**

```bash
# 客户端导出（推荐）
cd Configs/GameConfig && ./gen_code_bin_to_project.bat

# 服务端导出
cd Configs/GameConfig && ./gen_code_bin_to_server.bat
```

## 生成代码输出（Luban 自动生成）

| 输出类型 | 项目路径 |
|---------|---------|
| C# 代码 | `UnityProject/Assets/GameScripts/HotFix/GameProto/GameConfig/` |
| 二进制数据 | `UnityProject/Assets/AssetRaw/Configs/bytes/` |
| 桥接文件 | `UnityProject/Assets/GameScripts/HotFix/GameProto/ConfigSystem.cs` |

## Excel 文件说明

由于 .xlsx 是二进制格式，本示例用 Markdown 表格展示 Excel 内容结构：

- `__tables__.xlsx` — 查看表格描述了解如何注册新表
- `__beans__.xlsx` — 查看表格描述了解如何定义复合类型
- `__enums__.xlsx` — 查看表格描述了解如何定义枚举
- `item.xlsx` — 查看表格描述了解数据表格式
- `skill.xlsx` — 查看表格描述了解带 Bean 引用的复杂数据表
