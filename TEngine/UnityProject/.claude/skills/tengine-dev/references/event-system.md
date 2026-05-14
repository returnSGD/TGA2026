# 事件系统

> **适用场景**：GameEvent/AddUIEvent/GameEventMgr 使用、接口事件定义与发送、事件监听清理 | **关联文档**：[ui-lifecycle.md](ui-lifecycle.md)（AddUIEvent 自动清理）、[event-antipatterns.md](event-antipatterns.md)（避坑）、[naming-rules.md](naming-rules.md)（事件命名）

## 架构概览

TEngine 事件系统由三个核心组件构成：

| 组件 | 类型 | 职责 |
|------|------|------|
| **GameEvent** | 全局静态门面 | 持有 `static readonly EventMgr _eventMgr`，所有方法委托给内部 EventMgr 实例 |
| **GameEventMgr** | 局部作用域管理器 | 实现 `IMemory`，用于 UI 面板等需要随生命周期自动解绑的场景，只有 `AddEvent` + `Clear()` |
| **GameEventHelper** | Source Generator 生成类 | 源码中无 .cs 文件，由 `[EventInterface]` 特性在编译时自动生成，提供 `Init()` 和 `RegisterListener<T>()` |

TEngine 提供两种事件模式：**int/string 事件**（委托回调）和**接口事件**（类型安全）。

### 模式对比

| 特性 | int/string 事件 | 接口事件 |
|------|----------------|---------|
| 定义方式 | `const int` 常量 / string | 带 `[EventInterface]` 的接口 |
| 发送 | `GameEvent.Send(int/string)` | `GameEvent.Get<ITrade>().OnTrade(...)` |
| 监听 | `GameEvent.AddEventListener(int/string, callback)` | 实现接口 + `RegisterListener` |
| 类型安全 | 无编译检查 | 编译期检查 |
| 适用场景 | 简单通知、UI 内部 | 模块间通信、多参数 |

---

## 核心 API

### GameEvent 静态方法

#### Send（发送事件）

```csharp
// int 版本：支持 0~6 个泛型参数
GameEvent.Send(int eventType);
GameEvent.Send<T1>(int eventType, T1 arg1);
GameEvent.Send<T1,T2>(int eventType, T1 arg1, T2 arg2);
// ... 最多 Send<T1,T2,T3,T4,T5,T6>

// string 版本：支持 0~5 个泛型参数（内部通过 RuntimeId.ToRuntimeId 转为 int）
GameEvent.Send(string eventType);
GameEvent.Send<T1>(string eventType, T1 arg1);
// ... 最多 Send<T1,T2,T3,T4,T5>
```

#### AddEventListener（监听事件）

返回 `bool`（是否监听成功）。UI 内推荐用 `AddUIEvent`（自动清理，无需关心返回值）。

```csharp
// int 版本：支持 0~6 个泛型参数
bool GameEvent.AddEventListener(int eventType, Action handler);
bool GameEvent.AddEventListener<T1>(int eventType, Action<T1> handler);
// ... 最多 AddEventListener<T1,T2,T3,T4,T5,T6>

// string 版本：支持 0~5 个泛型参数
bool GameEvent.AddEventListener(string eventType, Action handler);
bool GameEvent.AddEventListener<T1>(string eventType, Action<T1> handler);
// ... 最多 AddEventListener<T1,T2,T3,T4,T5>
```

#### RemoveEventListener（移除监听）

```csharp
// int 版本：支持 0~5 个泛型参数 + Delegate 重载
GameEvent.RemoveEventListener(int eventType, Action handler);
GameEvent.RemoveEventListener<T1>(int eventType, Action<T1> handler);
// ... 最多 RemoveEventListener<T1,T2,T3,T4,T5>
GameEvent.RemoveEventListener(int eventType, Delegate handler);  // Delegate 重载

// string 版本：支持 0~5 个泛型参数 + Delegate 重载
GameEvent.RemoveEventListener(string eventType, Action handler);
// ... 最多 RemoveEventListener<T1,T2,T3,T4,T5>
GameEvent.RemoveEventListener(string eventType, Delegate handler);  // Delegate 重载
```

#### Get（接口事件获取）

```csharp
// 返回接口实例，内部调用 _eventMgr.GetInterface<T>()
T GameEvent.Get<T>();
```

#### Shutdown（清除所有事件注册）

```csharp
// 仅在游戏退出时调用，内部调用 _eventMgr.Init() 重置所有事件
GameEvent.Shutdown();
```

> **注意**：源码中没有 `UnRegisterAll<T>()` 或 `UnRegisterAll()` 方法。需要清除所有事件时使用 `Shutdown()`（全局）或 `GameEventMgr.Clear()`（局部）。

### GameEventMgr 局部管理器

实现 `IMemory`，仅支持 `int` eventType，最多 5 个泛型参数。没有 `RemoveEvent` 方法，通过 `Clear()` 一次性移除所有已注册事件。

```csharp
private readonly GameEventMgr _eventMgr = new();

// 注册（仅在 AddEventListener 返回 true 时才记录到内部列表）
_eventMgr.AddEvent(int eventType, Action handler);
_eventMgr.AddEvent<T1>(int eventType, Action<T1> handler);
// ... 最多 AddEvent<T1,T2,T3,T4,T5>

// 一次性移除所有
_eventMgr.Clear();
```

---

## 使用模式

### int 事件

```csharp
// 1. 定义事件接口（必须 [EventInterface]，需指定事件组）
[EventInterface(EEventGroup.GroupUI)]
public interface IGameEvent
{
    void OnGoldChanged();

    void OnHpChanged(int hp)
}

// 2. 源代码生成器自动实现并注册
public class IGameEvent_Event
{
    public static readonly int OnGoldChanged = RuntimeId.ToRuntimeId("IGameEvent_Event.OnGoldChanged");
    public static readonly int OnHpChanged = RuntimeId.ToRuntimeId("IGameEvent_Event.OnHpChanged");
}

public class IGameEvent_Gen : IGameEvent
{
    void OnGoldChanged() { /* 自动生成 */ }

    void OnHpChanged(int hp) { /* 自动生成 */ }
}

// 发送
GameEvent.Get<IGameEvent>().OnGoldChanged();
GameEvent.Get<IGameEvent>().OnHpChanged(hp);

// 监听（UI 内用 AddUIEvent 自动清理）
AddUIEvent(IGameEvent_Event.OnGoldChanged, OnGoldChanged);
AddUIEvent<int>(IGameEvent_Event.OnHpChanged, OnHpChanged);

// 非 UI 类监听（必须手动移除）
void OnEnable() { GameEvent.AddEventListener<int>(IGameEvent_Event.OnHpChanged, OnHpChanged); }
void OnDisable() { GameEvent.RemoveEventListener<int>(IGameEvent_Event.OnHpChanged, OnHpChanged); }
```

### string 事件类型

除 `int` 外，`GameEvent` 也支持 `string` 作为事件 ID（API 与 int 版本对称，但是不推荐使用）：

```csharp
// 发送
GameEvent.Send("OnGoldChanged");
GameEvent.Send<int>("OnHpChanged", 50);

// 监听（UI 内）
AddUIEvent("OnGoldChanged", OnGoldChanged);
AddUIEvent<int>("OnHpChanged", OnHpChanged);

// 非 UI 类监听
GameEvent.AddEventListener<int>("OnHpChanged", OnHpChanged);
GameEvent.RemoveEventListener<int>("OnHpChanged", OnHpChanged);
```

适用场景：事件名需要动态拼接、或跨模块字符串约定时使用。性能略低于 int（内部通过 `RuntimeId.ToRuntimeId` 转换），优先用 int。

### 接口事件

```csharp
// 1. 定义接口（必须 [EventInterface]，需指定事件组）
[EventInterface(EEventGroup.GroupUI)]
public interface ITrade
{
    void OnTradeComplete(int itemId, int count);
}

// 2. 源代码生成器自动实现并注册
public class ITrade_Gen : ITrade
{
    void OnTradeComplete(int itemId, int count) { /* 自动生成 */ }
}

// 3. 发送
GameEvent.Get<ITrade>().OnTradeComplete(itemId, count);
```

**前提**：`GameEventHelper.Init()` 已在 `GameApp.Entrance` 中最先调用。

> **注意**：`GameEventHelper` 是 Source Generator 自动生成的类，源码中不存在 `.cs` 文件。编译时由事件接口上的 `[EventInterface]` 特性触发生成，提供 `Init()`（注册所有接口事件）和 `RegisterListener<T>()`（运行时注册实现类）方法。

### GameEventMgr 批量管理

非 UI 类的事件监听推荐用 `GameEventMgr` 统一管理，避免忘记移除：

```csharp
private readonly GameEventMgr _eventMgr = new();

public void Init()
{
    _eventMgr.AddEvent(GameEventDef.OnGoldChanged, OnGoldChanged);
    _eventMgr.AddEvent<int>(GameEventDef.OnHpChanged, OnHpChanged);
}

public void Dispose() => _eventMgr.Clear();  // 一次性移除所有
```

---

## 常见错误

### 1. 忘记 GameEventHelper.Init()

```csharp
// 错误：GameEvent.Get<T>() 全部无响应，无报错，极难排查
public static void Entrance(Assembly[] assemblies)
{
    // GameEventHelper.Init();  <- 忘记调用
    GameApp_RegisterSystem.Register();
}

// 正确：必须最先调用
GameEventHelper.Init();
```

### 2. UI 外部使用 AddEventListener（内存泄漏）

```csharp
// 错误：退出窗口不会自动清理
void SomeMethod()
    => GameEvent.AddEventListener<int>(IBattleEvent_Event.OnHpChanged, OnHpChanged);

// 正确：UIWindow 中用 AddUIEvent（自动清理）
protected override void RegisterEvent()
    => AddUIEvent<int>(IBattleEvent_Event.OnHpChanged, OnHpChanged);

// 正确：非 UI 类用 GameEventMgr
private readonly GameEventMgr _eventMgr = new();
public void Dispose() => _eventMgr.Clear();
```

### 3. 手写事件 ID 常量

```csharp
// 错误：手写 int 常量，容易重复/拼错
public const int OnHpChanged = 1001;

// 正确：Source Generator 自动生成
AddUIEvent(IBattleEvent_Event.OnHpChanged, OnHpChanged);
```

### 4. 事件回调签名不匹配

```csharp
// 错误：事件发送 int，回调接收 string -> 运行时异常
GameEvent.Send<int>(IBattleEvent_Event.OnHpChanged, hp);
AddUIEvent<string>(IBattleEvent_Event.OnHpChanged, OnHp);

// 正确：接口事件模式可编译期检查
GameEvent.Get<IBattleEvent>().OnHpChanged(hp); // 类型安全
```

### 5. 非 UI 类忘记移除监听

```csharp
// 错误：销毁时不移除，回调引用已释放对象
public class PlayerSystem
{
    public void Init() => GameEvent.AddEventListener(IPlayerEvent_Event.OnDead, OnDead);
    // 没有 RemoveEventListener -> 泄漏
}

// 正确：使用 GameEventMgr 批量清理
private readonly GameEventMgr _eventMgr = new();
public void Init()    => _eventMgr.AddEvent(IPlayerEvent_Event.OnDead, OnDead);
public void Dispose() => _eventMgr.Clear();
```

### 6. 误用不存在的 UnRegisterAll

```csharp
// 错误：源码中不存在 UnRegisterAll<T>() 或 UnRegisterAll() 方法
GameEvent.UnRegisterAll<int>(eventType);
GameEvent.UnRegisterAll();

// 正确：按需使用以下方式清除
GameEvent.RemoveEventListener(eventType, handler);  // 移除单个监听
GameEventMgr.Clear();                               // 局部批量清除
GameEvent.Shutdown();                               // 全局清除（仅游戏退出时）
```

### 7. 误用不存在的 RegisterListener

```csharp
// 错误：GameEvent 中不存在 RegisterListener<T>() 方法
GameEvent.RegisterListener<ITrade>(implementation);

// 正确：GameEventHelper.Init() 由 Source Generator 自动注册，无需手动调用 RegisterListener
// 接口事件实现类由编译时自动生成和注册，只需确保 GameEventHelper.Init() 在 GameApp.Entrance 中最先调用
GameEventHelper.Init();
```

---

## 事件定义规范

| 规则 | 说明 |
|------|------|
| ID 范围 | 自定义事件 >= 10000，1~9999 保留 |
| 命名 | `On` + 过去式动词 + 名词：`OnGoldChanged`、`OnBattleEnded` |
| 接口命名 | `I` + 动词 + 名词：`ITrade`、`IBattle` |
| 泛型参数 | int 事件最多 6 个，string 事件最多 5 个，GameEventMgr 最多 5 个 |
| 禁止 | 手写事件 ID 硬编码数字，应用常量 |

---

## 交叉引用

| 相关文档 | 内容 |
|---------|------|
| ui-lifecycle.md | AddUIEvent 在 UIWindow 生命周期中的自动清理机制 |
| modules.md | GameModule 事件相关模块 |
| event-antipatterns.md | 事件系统反模式与避坑指南 |
| naming-rules.md | 事件常量与接口的命名约定 |
