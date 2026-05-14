# TEngine 架构与项目结构

> **适用场景**：理解项目分层架构、程序集划分、目录结构、启动流程 | **关联文档**：[modules.md](modules.md)、[hotfix-workflow.md](hotfix-workflow.md)、[resource-api.md](resource-api.md)

## 核心架构

### 分层架构

```
┌─────────────────────────────────────────────┐
│         游戏业务层 (HotFix)                  │
│  GameLogic / GameProto                      │
│  ↑ 热更代码，业务逻辑，高频变更              │
└─────────────────────────────────────────────┘
                     ↓ 依赖
┌─────────────────────────────────────────────┐
│         框架应用层 (Launcher + Assembly-CSharp)│
│  流程管理 / 热更新加载 / 启动器 / UI加载器   │
│  ↑ 主包代码，不参与热更                      │
└─────────────────────────────────────────────┘
                     ↓ 依赖
┌─────────────────────────────────────────────┐
│         框架核心层 (TEngine)                 │
│  ModuleSystem / EventSystem / ResourceMgr   │
│  ↑ TEngine.Runtime 程序集                   │
└─────────────────────────────────────────────┘
                     ↓ 依赖
┌─────────────────────────────────────────────┐
│         基础设施层 (Unity + 第三方)           │
│  YooAsset / HybridCLR / UniTask / Luban     │
└─────────────────────────────────────────────┘
```

**依赖规则**：单向向下依赖，上层不能被下层引用，热更代码不能反向依赖主工程 internal 类型。

---

### 程序集划分

| 程序集 | 路径 | 热更 | 职责 |
|--------|------|------|------|
| `TEngine.Runtime` | Assets/TEngine/Runtime/ | 否 | 框架核心 |
| `TEngine.Editor` | Assets/TEngine/Editor/ | 否 | 编辑器工具 |
| `Launcher` | Assets/Launcher/ | 否 | 启动器 UI（LoadUpdateUI、LoadTipsUI 等） |
| Assembly-CSharp | Assets/GameScripts/ | 否 | GameEntry 启动入口、主包流程 |
| `GameProto` | Assets/GameScripts/HotFix/GameProto/ | 是 | Luban 配置代码 |
| `GameLogic` | Assets/GameScripts/HotFix/GameLogic/ | 是 | 业务逻辑，热更主入口 |

**约束**：
- `GameLogic` 可依赖 `GameProto` 和 `TEngine.Runtime`
- 热更代码不能引用主工程 internal 类型
- `GameEntry.cs` 和 `Procedure/` 属于 Assembly-CSharp，不参与热更

---

### 目录结构

```
TEngine/
├── Configs/GameConfig/           # Luban 配置表工程
│   ├── Datas/                    # Excel 数据源
│   └── gen_code_bin_to_project.bat
│
└── UnityProject/Assets/
    ├── AssetRaw/                 # 热更资源目录（YooAsset 打包来源）
    │   ├── Actor/ Audios/ Configs/ Effects/ Fonts/
    │   ├── Materials/ Scenes/ Shaders/ UI/
    │   └── ...
    ├── Launcher/                 # 启动器模块
    │   └── Scripts/              # LauncherMgr、加载UI 等
    ├── TEngine/                  # 框架核心代码
    │   ├── Editor/               # 编辑器工具
    │   └── Runtime/              # 运行时核心（Core/ Extension/ Module/）
    └── GameScripts/              # 游戏逻辑脚本
        ├── GameEntry.cs          # 游戏启动入口
        ├── Procedure/            # 主包流程状态机
        └── HotFix/               # 热更代码
            ├── GameProto/        # Luban 生成代码
            └── GameLogic/        # 业务逻辑主开发区域
```

---

### 启动流程

```
GameEntry.Awake()
  ├── ModuleSystem.GetModule<IUpdateDriver>()    // 帧驱动
  ├── ModuleSystem.GetModule<IResourceModule>()  // 资源系统
  ├── ModuleSystem.GetModule<IDebuggerModule>()  // 调试器
  ├── ModuleSystem.GetModule<IFsmModule>()       // 状态机
  ├── DontDestroyOnLoad(this)
  └── Settings.ProcedureSetting.StartProcedure().Forget()  // 启动流程

主包流程（不可热更）：
ProcedureLaunch → ProcedureSplash → ProcedureInitResources → ProcedureInitPackage
→ ProcedureCreateDownloader → ProcedureDownloadFile → ProcedurePreload
→ ProcedureLoadAssembly → ProcedureStartGame

热更入口：ProcedureLoadAssembly 加载完热更 DLL 后通过反射调用
  GameApp.Entrance(object[] objects)
  ├── GameEventHelper.Init()          // 1. 必须最先调用
  ├── 保存热更程序集列表              // 2. objects[0] 为 List<Assembly>
  ├── Utility.Unity.AddDestroyListener(Release)  // 3. 注册销毁回调
  └── StartGameLogic()                // 4. 启动游戏逻辑
```

---

## 使用模式

### 资源目录组织

```
Assets/AssetRaw/              # 所有热更资源的根目录
├── Actor/                    # 角色 Prefab
├── Audios/BGM/ SFX/          # 音频
├── Configs/bytes/            # Luban 生成数据
├── Effects/                  # 粒子特效
├── Scenes/                   # 场景
├── UI/Atlas/ Prefabs/        # UI 资源
└── ...
```

- **PRELOAD** 标签：启动时预加载（配置数据、公共 UI）
- 资源 location 等于文件名（不含路径和扩展名），YooAsset 自动收集
- **禁止** `Resources.Load()`，所有资源通过 `AssetRaw/` + YooAsset 管理

---

## 常见错误

| 错误 | 原因 | 修复 |
|------|------|------|
| 误认为 GameScripts.Main 程序集存在 | GameEntry.cs 和 Procedure/ 无自定义 asmdef | 它们属于 Assembly-CSharp |
| 热更入口签名写错 | 文档写 `Entrance(Assembly[])` | 实际为 `Entrance(object[])`，objects[0] 强转为 List<Assembly> |
| 流程链遗漏 ProcedureSplash | 直接从 ProcedureLaunch 跳到 ProcedureInitResources | ProcedureLaunch → ProcedureSplash 才是正确的下一步 |

---

## 交叉引用

- 模块 API 见 [modules.md](modules.md)
- 热更开发见 [hotfix-workflow.md](hotfix-workflow.md)
- 事件系统见 [event-system.md](event-system.md)
- 资源加载见 [resource-api.md](resource-api.md)
