# 热更代码开发与热更包管理

> **适用场景**：HybridCLR 热更边界划分、GameApp 热更入口、DLL 加载流程、AOT 泛型补全 | **关联文档**：[architecture.md](architecture.md)（程序集划分）、[modules.md](modules.md)（GameModule 访问）

## 核心 API

### 程序集划分

```
GameScripts/HotFix/
├── GameProto/          # Luban 生成（勿手改）
│   ├── LubanLib/       # ByteBuf 等序列化库
│   ├── GameConfig/     # Tables + 配置类
│   └── ConfigSystem.cs # 配置加载器（需从 Luban 模板生成）
│
└── GameLogic/          # 业务逻辑（主开发区域）
    ├── GameApp.cs                  # 热更主入口（partial class）
    ├── GameModule.cs               # 模块统一访问入口
    ├── IEvent/                     # 事件接口定义
    ├── Module/                     # 模块实现（如 UIModule）
    ├── SingletonSystem/            # 单例系统
    ├── UI/                         # UI 窗口代码
    └── ...
```

**依赖规则**（不可逆向）：`GameLogic → GameProto`、`GameLogic → TEngine.Runtime`
主包 `GameScripts/` + `Launcher/` 不可热更，仅含启动器和流程驱动。

---

### GameApp 入口

```csharp
// GameApp.cs（partial class）
public partial class GameApp
{
    private static List<Assembly> _hotfixAssembly;

    /// <summary>
    /// 热更域App主入口。由 ProcedureLoadAssembly 通过反射调用。
    /// </summary>
    public static void Entrance(object[] objects)
    {
        GameEventHelper.Init();                                  // 1. 必须最先调用
        _hotfixAssembly = (List<Assembly>)objects[0];            // 2. 保存热更程序集
        Utility.Unity.AddDestroyListener(Release);               // 3. 注册销毁回调
        StartGameLogic();                                        // 4. 启动游戏逻辑
    }

    private static void StartGameLogic()
    {
        GameModule.UI.ShowUIAsync<BattleMainUI>();               // 示例：直接打开 UI
    }

    private static void Release()
    {
        SingletonSystem.Release();                               // 释放单例
    }
}
```

**关键约束**：
- `GameEventHelper.Init()` 必须在任何 `GameEvent.Get<T>()` 之前调用
- `Entrance` 参数类型是 `object[]`，不是 `Assembly[]`——由 `ProcedureLoadAssembly` 反射调用时传入
- 可通过 `partial class GameApp` 拓展入口逻辑（如注册系统）

---

### 流程状态机

热更域流程继承 `ProcedureBase`：

```csharp
public class ProcedureLogin : ProcedureBase
{
    protected override void OnEnter(ProcedureOwner owner)
        => GameModule.UI.ShowUIAsync<LoginUI>();
    protected override void OnLeave(ProcedureOwner owner, bool isShutdown)
        => GameModule.UI.CloseUI<LoginUI>();
}

// 流程切换
ChangeState<ProcedureMain>(procedureOwner);            // 内部
GameModule.Procedure.ChangeState<ProcedureMain>();      // 外部
```

---

### HybridCLR 注意事项

```csharp
// ❌ 热更代码不能引用主包 internal 类型
// ❌ 避免 System.Reflection 大量使用（AOT 限制多）
// ❌ 不支持：dynamic、Expression<T> 编译、部分 Emit、Marshal/P-Invoke
```

#### AOT 泛型补充

```
HybridCLR → Generate → AOT Generic References → 生成 AOTGenericReferences.cs
如出现 ExecutionEngineException，手动添加对应泛型使用后重新打包
```

常见需补充：`List<自定义类型>`、`Dictionary<K,V>` 新组合、`UniTask<自定义类型>`、`Action<自定义类型>`

---

## 使用模式

### 热更包下载流程

主包流程驱动，不可热更：

```
ProcedureLaunch → ProcedureSplash → ProcedureInitResources → ProcedureInitPackage
→ ProcedureCreateDownloader → ProcedureDownloadFile → ProcedurePreload
→ ProcedureLoadAssembly → ProcedureStartGame
```

#### 核心代码

```csharp
public async UniTask CheckAndDownloadUpdate()
{
    // 1. 请求远端版本
    var versionOp = GameModule.Resource.RequestPackageVersionAsync();
    await versionOp.Task;
    // 2. 更新 Manifest
    var manifestOp = GameModule.Resource.UpdatePackageManifestAsync(versionOp.PackageVersion);
    await manifestOp.Task;
    // 3. 创建下载器
    var downloader = GameModule.Resource.CreateResourceDownloader();
    if (downloader.TotalDownloadCount == 0) return;
    // 4. 下载
    downloader.OnDownloadProgressCallback = OnProgress;
    downloader.BeginDownload();
    await downloader.Task;
    // 5. 清理旧缓存
    await GameModule.Resource.ClearCacheFilesAsync();
}
```

#### API 速查

| 方法 | 说明 |
|------|------|
| `GetPackageVersion()` | 本地资源包版本号 |
| `RequestPackageVersionAsync()` | 请求远端版本号 |
| `UpdatePackageManifestAsync(ver)` | 更新资源清单 |
| `CreateResourceDownloader()` | 创建差量下载器 |
| `downloader.TotalDownloadCount` | 待下载文件数 |
| `downloader.TotalDownloadBytes` | 待下载总字节 |
| `ClearCacheFilesAsync()` | 清理冗余缓存 |

---

### 日常开发步骤

```
1. 在 GameScripts/HotFix/ 下修改/添加代码
2. Editor 模式：直接 Play（热更 DLL 编译进 Assets）
3. 模拟热更：HybridCLR → Generate All → 打资源包 → 拷贝 DLL
4. 真机测试：出包 → 部署热更资源到 CDN → 启动触发热更
```

#### 新功能开发

```
1. IEvent/ 定义事件接口（跨模块通信时）
2. UI/ 创建 UIWindow 子类（[Window] 特性）
3. Module/ 实现业务模块
4. partial class GameApp 拓展入口初始化逻辑
5. Procedure 中连接 UI 打开与系统初始化
```

---

## 常见错误

| 错误 | 原因 | 修复 |
|------|------|------|
| `GameApp_RegisterSystem` 不存在 | 文档描述了不存在的文件 | GameLogic 下无此文件，使用 partial class GameApp 拓展 |
| `Entrance(Assembly[])` 签名错误 | 文档与源码不一致 | 实际为 `Entrance(object[])`，objects[0] 是 `List<Assembly>` |
| 热更代码找不到 GameApp.Entrance | 主包反射调用失败 | 确认 `UpdateSetting` 中 `LogicMainDllName` 指向 GameLogic.dll |
| 流程链错误 | 文档遗漏 ProcedureSplash | 正确链路从 ProcedureLaunch 开始，经过 ProcedureSplash |
| 热更入口中直接 ChangeState | 文档描述了 ProcedureLogin 流程切换 | 当前 GameApp.StartGameLogic 直接打开 UI，无额外流程 |

---

## 交叉引用

- 架构总览见 [architecture.md](architecture.md)
- 事件系统见 [event-system.md](event-system.md)
- 资源加载见 [resource-api.md](resource-api.md)
- UI 生命周期见 [ui-lifecycle.md](ui-lifecycle.md)
