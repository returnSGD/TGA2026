# TEngine 模块 API 速查

> **适用场景**：使用 GameModule.Timer/Scene/Audio/Fsm/MemoryPool/Log 等模块 API | **关联文档**：[ui-lifecycle.md](ui-lifecycle.md)（UI 模块）、[resource-api.md](resource-api.md)（Resource 模块）、[event-system.md](event-system.md)（事件模块）

## 核心 API：GameModule 统一访问入口

所有模块通过 `GameModule` 静态类访问（已缓存），禁止重复 `ModuleSystem.GetModule<T>()`：

```csharp
GameModule.Base          // RootModule          — 根模块（框架初始化入口）
GameModule.Debugger      // IDebuggerModule     — 调试器（`~` 键呼出）
GameModule.Fsm           // IFsmModule          — 有限状态机
GameModule.Procedure     // IProcedureModule    — 流程
GameModule.Resource      // IResourceModule     — 资源加载
GameModule.Audio         // IAudioModule        — 音频
GameModule.UI            // UIModule            — UI 管理
GameModule.Scene         // ISceneModule        — 场景
GameModule.Timer         // ITimerModule        — 计时器
GameModule.Localization  // ILocalizationModule — 本地化

GameModule.Shutdown()   // 清空所有模块缓存引用，仅在游戏退出时调用
```

> **注意**：`UI` 属性类型是 `UIModule`（单例），不是 `IUIModule`；`Base` 属性通过 `FindObjectOfType<RootModule>()` 获取，其余通过 `ModuleSystem.GetModule<T>()` 获取。

---

## 使用模式

### TimerModule 计时器

```csharp
// 添加计时器
int tid = GameModule.Timer.AddTimer(OnTick, time: 3f);                         // 单次
int tid = GameModule.Timer.AddTimer(OnTick, time: 1f, isLoop: true);           // 循环
int tid = GameModule.Timer.AddTimer(OnTick, time: 5f, isUnscaled: true);       // 非缩放时间

// 控制
GameModule.Timer.Stop(timerId);         // 暂停
GameModule.Timer.Resume(timerId);       // 恢复
GameModule.Timer.Restart(timerId);      // 重置
GameModule.Timer.RemoveTimer(timerId);  // 移除（销毁时必须调用）
GameModule.Timer.RemoveAllTimer();      // 清除所有

// 查询
float left = GameModule.Timer.GetLeftTime(timerId);
bool running = GameModule.Timer.IsRunning(timerId);
```

### SceneModule 场景管理

```csharp
// 加载
Scene scene = await GameModule.Scene.LoadSceneAsync("SceneName");
Scene scene = await GameModule.Scene.LoadSceneAsync("SceneName", LoadSceneMode.Additive);
Scene scene = await GameModule.Scene.LoadSceneAsync("SceneName", LoadSceneMode.Single,
    progressCallBack: p => { /* 0~1 */ });

// 卸载/激活
bool ok = await GameModule.Scene.UnloadAsync("SceneName");
GameModule.Scene.ActivateScene("SceneName");
bool has = GameModule.Scene.IsContainScene("SceneName");
```

### AudioModule 音频

```csharp
// 播放
AudioAgent agent = GameModule.Audio.Play(AudioType.Music, "bgm_path", bLoop: true);   // BGM
AudioAgent agent = GameModule.Audio.Play(AudioType.Sound, "sfx_path");                  // 音效
AudioAgent agent = GameModule.Audio.Play(AudioType.UISound, "ui_click", bAsync: true);  // UI

// 停止
GameModule.Audio.Stop(AudioType.Music, fadeout: true);
GameModule.Audio.StopAll(fadeout: false);

// 音量
GameModule.Audio.Volume      = 1.0f;  // 全局 (0~1)
GameModule.Audio.MusicVolume = 0.8f;
GameModule.Audio.SoundVolume = 1.0f;
GameModule.Audio.MusicEnable = true;
GameModule.Audio.SoundEnable = true;
```

### FsmModule 有限状态机

```csharp
// 定义状态
public class IdleState : FsmState<MyOwner>
{
    protected override void OnEnter(IFsm<MyOwner> fsm)  { }
    protected override void OnUpdate(IFsm<MyOwner> fsm, float elapse, float real) { }
    protected override void OnLeave(IFsm<MyOwner> fsm, bool isShutdown) { }
}

// 创建并启动
IFsm<MyOwner> fsm = GameModule.Fsm.CreateFsm<MyOwner>("FsmName", owner,
    new IdleState(), new RunState(), new AttackState());
fsm.Start<IdleState>();

// 切换与传数据
fsm.ChangeState<RunState>();
fsm.SetData<int>("Key", value);
int val = fsm.GetData<int>("Key");

// 销毁
GameModule.Fsm.DestroyFsm<MyOwner>("FsmName");
```

### MemoryPool 内存池

频繁创建/销毁的纯 C# 对象，避免 GC：

```csharp
public class DamageInfo : IMemory
{
    public int Damage;
    public void Clear() { Damage = 0; }  // 归还时重置
}

var info = MemoryPool.Acquire<DamageInfo>();
info.Damage = 100;
MemoryPool.Release(info);  // Release 后禁止再访问，禁止 Release 两次
```

### Log 日志系统

```csharp
Log.Debug("仅 Development Build 输出");  // 发布包自动剥离
Log.Info("普通信息");
Log.Warning("警告");
Log.Error("错误，始终保留");
Log.Fatal("严重错误");
Log.Assert(condition, "断言失败提示");
```

---

## 常见错误

| 错误写法 | 正确写法 | 原因 |
|---------|---------|------|
| `ModuleSystem.GetModule<ITimerModule>()` | `GameModule.Timer` | 重复查找，未利用缓存 |
| `OnDestroy` 忘记 `RemoveTimer(tid)` | 必须调用 `RemoveTimer` | 计时器回调引用已销毁对象，导致空引用 |
| `SceneManager.LoadScene()` | `GameModule.Scene.LoadSceneAsync()` | 绕过框架资源管理，热更包无法加载 |
| `GameModule.UI` 误用接口 `IUIModule` | 类型是 `UIModule`（单例实现） | 源码中 UI 属性返回 `UIModule.Instance`，非 `GetModule<T>()` |
| `Shutdown()` 后继续访问模块属性 | `Shutdown()` 仅游戏退出时调用 | 所有缓存引用置 null，后续访问触发重新查找或空引用 |
| `MemoryPool.Release()` 后访问对象 | Release 后禁止再访问 | 对象已归还池中，状态不确定 |
| `MemoryPool.Release()` 同一对象两次 | 确保只 Release 一次 | 重复归还导致池状态异常 |
| `GameModule.LoadScene` | `GameModule.Scene.LoadSceneAsync` | 不存在 `GameModule.LoadScene`，场景加载通过 `GameModule.Scene` |
| `new FsmState<>()` | 继承 `FsmState<TOwner>` | 状态必须继承基类，不能直接 new |
| `GameModule.Timer.AddTimer(time, callback)` | `GameModule.Timer.AddTimer(callback, time)` | 参数顺序：回调在前，时间在后 |

---

## 交叉引用

| 关联主题 | 文档 | 说明 |
|---------|------|------|
| 资源加载/卸载 | resource-api.md | `GameModule.Resource` 的完整 API 与生命周期 |
| UI 管理 | ui-lifecycle.md | `GameModule.UI` 的窗口生命周期与层级 |
| UI 进阶 | ui-patterns.md | Widget 模板与节点绑定 |
| 事件系统 | event-system.md | `GameEvent` 模块间解耦，`AddUIEvent` UI 内部事件 |
| 热更边界 | hotfix-workflow.md | `GameModule` 所在程序集与热更边界 |
| 资源管理模式 | resource-patterns.md | 资源生命周期与模块协作 |
