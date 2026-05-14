# 事件系统反模式与避坑指南

> **适用场景**：事件内存泄漏排查、接口事件无响应调试、事件风暴问题定位 | **关联文档**：[event-system.md](event-system.md)（核心 API）、[ui-lifecycle.md](ui-lifecycle.md)（AddUIEvent 自动清理）

> 本文档是 [event-system.md](event-system.md) 的进阶补充，聚焦于难以排查的陷阱和反模式。

---

## 一、内存泄漏反模式

### 反模式 1：UIWindow 外部直接 AddEventListener

```csharp
// ❌ 错误：退出窗口后监听不清理，回调引用已销毁的对象
public class BagUI : UIWindow
{
    protected override void OnCreate()
    {
        // 在 OnCreate 而非 RegisterEvent 中注册，且用了全局方法
        GameEvent.AddEventListener<int>(IItemEvent_Event.OnItemChanged, OnItemChanged);
    }
}

// ✅ 正确：RegisterEvent 内使用 AddUIEvent，随窗口销毁自动清理
public class BagUI : UIWindow
{
    protected override void RegisterEvent()
    {
        AddUIEvent<int>(IItemEvent_Event.OnItemChanged, OnItemChanged);
    }
}
```

**为何危险**：`AddUIEvent` 在 `RemoveAllUIEvent()` 时批量清理（UIWindow.OnDestroy 自动调用）。`GameEvent.AddEventListener` 不会自动清理，窗口销毁后仍持有回调引用，导致访问已销毁 GameObject。

---

### 反模式 2：非 UI 类不清理监听

```csharp
// ❌ 错误：系统类注册事件但从不移除
public class PlayerSystem
{
    public void Init()
    {
        GameEvent.AddEventListener(IPlayerEvent_Event.OnLevelUp, OnLevelUp);
        GameEvent.AddEventListener<int>(IPlayerEvent_Event.OnHpChanged, OnHpChanged);
        // 没有对应的 Remove
    }
    // OnDisable/OnDestroy 中也没有清理
}

// ✅ 正确：用 GameEventMgr 统一管理，Dispose 时一次清理
public class PlayerSystem
{
    private readonly GameEventMgr _eventMgr = new();

    public void Init()
    {
        _eventMgr.AddEvent(IPlayerEvent_Event.OnLevelUp, OnLevelUp);
        _eventMgr.AddEvent<int>(IPlayerEvent_Event.OnHpChanged, OnHpChanged);
    }

    public void Dispose() => _eventMgr.Clear();
}
```

---

### 反模式 3：Lambda 捕获导致泄漏

```csharp
// ❌ 错误：Lambda 无法用 RemoveEventListener 移除（引用不同）
public void Init()
{
    GameEvent.AddEventListener<int>(eventId, hp => { _textHp.text = hp.ToString(); });
    // 后续 RemoveEventListener 传入新 Lambda，无法匹配
}

// ✅ 正确：命名方法，可精确移除；或改用 GameEventMgr/AddUIEvent
private void OnHpChanged(int hp) => _textHp.text = hp.ToString();

public void Init()   => _eventMgr.AddEvent<int>(eventId, OnHpChanged);
public void Dispose() => _eventMgr.Clear();
```

---

## 二、接口事件无响应

### 反模式 4：遗忘 GameEventHelper.Init()

```csharp
// ❌ 错误：GameEvent.Get<T>() 调用无响应，无任何报错，极难排查
public static void Entrance(object[] objects)
{
    _hotfixAssembly = (List<Assembly>)objects[0];
    StartGameLogic();  // 此时所有接口事件都无法响应
}

// ✅ 正确：必须最先调用，在任何 GameEvent.Get<T>() 之前
public static void Entrance(object[] objects)
{
    GameEventHelper.Init();                       // 第一行
    _hotfixAssembly = (List<Assembly>)objects[0];
    Utility.Unity.AddDestroyListener(Release);
    StartGameLogic();
}
```

**排查方法**：若所有接口事件都无响应，首先确认 `GameEventHelper.Init()` 是否已调用。int/string 事件不受此影响（它们不依赖 Source Generator 初始化）。

---

### 反模式 5：接口缺少 [EventInterface] 特性

```csharp
// ❌ 错误：没有 [EventInterface]，Source Generator 不生成代码
public interface IBattleEvent
{
    void OnHpChanged(int hp);
}

// ✅ 正确：必须标注 [EventInterface] 并指定事件组
[EventInterface(EEventGroup.GroupBattle)]
public interface IBattleEvent
{
    void OnHpChanged(int hp);
}
```

**注意**：`EEventGroup` 枚举需在主项目中定义（或已有组使用），用于隔离不同模块的事件。

---

## 三、类型不匹配导致运行时异常

### 反模式 6：Send 与 AddEventListener 泛型参数不一致

```csharp
// ❌ 错误：发送 int，监听 float → 运行时 InvalidCastException
GameEvent.Send<int>(IPlayerEvent_Event.OnHpChanged, 100);
AddUIEvent<float>(IPlayerEvent_Event.OnHpChanged, OnHpChanged);

// ❌ 错误：发送 (int, string)，监听 (string, int) → 运行时异常
GameEvent.Send<int, string>(eventId, 1, "sword");
AddUIEvent<string, int>(eventId, OnItemAcquired);

// ✅ 正确：接口事件模式可在编译期发现此类问题
GameEvent.Get<IPlayerEvent>().OnHpChanged(100);  // 编译期类型检查
```

---

### 反模式 7：混用 int 和 string 事件 ID

```csharp
// ❌ 错误：注册用 int，发送用 string（它们走不同的 RuntimeId 路径）
GameEvent.AddEventListener<int>(1001, OnHpChanged);   // 直接 int
GameEvent.Send<int>("OnHpChanged", 50);               // string 转 RuntimeId

// 上面两行 ID 不同，监听收不到事件

// ✅ 正确：统一使用接口生成的常量
AddUIEvent<int>(IPlayerEvent_Event.OnHpChanged, OnHpChanged);
GameEvent.Send<int>(IPlayerEvent_Event.OnHpChanged, 50);
```

---

## 四、生命周期时序陷阱

### 反模式 8：在 OnCreate 之前访问事件

```csharp
// UIWindow 生命周期顺序：ScriptGenerator → RegisterEvent → OnCreate → OnRefresh
// RegisterEvent 在 OnCreate 之前执行，此时数据可能未就绪

// ❌ 错误：RegisterEvent 中直接访问需在 OnCreate 初始化的数据
protected override void RegisterEvent()
{
    AddUIEvent<int>(IPlayerEvent_Event.OnHpChanged, RefreshHp);
    RefreshHp(_currentHp);   // _currentHp 在 OnCreate 中初始化，此时为默认值
}

// ✅ 正确：RegisterEvent 只注册，OnRefresh 做刷新
protected override void RegisterEvent()
{
    AddUIEvent<int>(IPlayerEvent_Event.OnHpChanged, RefreshHp);
}

protected override void OnRefresh()
{
    RefreshHp(PlayerData.Hp);   // 每次 ShowUI 时刷新
}
```

---

### 反模式 9：在 Widget.OnDestroy 中访问父 Window 的事件

```csharp
// UIWindow 销毁顺序：RemoveAllUIEvent → 子Widget.OnDestroy → UIWindow.OnDestroy
// 子 Widget 销毁时父 Window 的 UIEvent 已经清理

// ❌ 错误：Widget 销毁时向父 Window 发送事件，但父 Window 已停止监听
public class ItemWidget : UIWidget
{
    protected override void OnDestroy()
    {
        GameEvent.Send(IItemEvent_Event.OnItemDestroyed);  // 发出去，但父 Window 已不再监听
    }
}

// ✅ 正确：Widget 间通信通过公开方法，或 Widget.OnDestroy 不依赖父 Window 事件
```

---

## 五、事件风暴反模式

### 反模式 10：事件回调中触发同类事件

```csharp
// ❌ 危险：OnGoldChanged 回调中又发送 OnGoldChanged → 无限递归
private void OnGoldChanged(int gold)
{
    _textGold.text = gold.ToString();
    if (gold > 1000)
    {
        GameEvent.Get<IPlayerEvent>().OnGoldChanged(gold - 100);  // 递归！
    }
}

// ✅ 正确：事件回调只更新 UI，业务逻辑由系统层控制
private void OnGoldChanged(int gold)
{
    _textGold.text = gold.ToString();
    // 不在回调中修改状态或触发新事件
}
```

---

### 反模式 11：高频事件直接更新 UI

```csharp
// ❌ 错误：每帧都发送 OnPositionChanged 并直接更新 UI，产生大量 UI 重绘
void Update()
{
    GameEvent.Get<IHeroEvent>().OnPositionChanged(_hero.position);
}

// ✅ 正确：高频数据用 Timer 轮询或 OnUpdate 节流
int _timerId;

protected override void OnCreate()
{
    _timerId = GameModule.Timer.AddTimer(RefreshPosition, time: 0.1f, isLoop: true);
}

protected override void OnDestroy()
{
    GameModule.Timer.RemoveTimer(_timerId);
}

private void RefreshPosition() => _textPos.text = _hero.position.ToString();
```

---

## 六、不存在的 API（AI 常见幻觉）

```csharp
// ❌ 以下 API 均不存在，编译失败：
GameEvent.UnRegisterAll();                    // 不存在
GameEvent.UnRegisterAll<int>(eventId);        // 不存在
GameEvent.RegisterListener<ITrade>(impl);     // 不存在（由 Source Generator 处理）
GameEvent.ClearAll();                         // 不存在
GameEvent.RemoveAll(eventId);                 // 不存在

// ✅ 正确替代：
GameEvent.RemoveEventListener(eventId, handler);   // 移除单个
GameEventMgr.Clear();                              // 局部批量清除
GameEvent.Shutdown();                              // 全局清除（仅游戏退出）
```

---

## 交叉引用

| 主题 | 文档 |
|------|------|
| 事件系统核心 API | [event-system.md](event-system.md) |
| UIWindow 生命周期与 AddUIEvent | [ui-lifecycle.md](ui-lifecycle.md) |
| Timer 替代高频事件 | [modules.md](modules.md) |
| 问题排查 | [troubleshooting.md](troubleshooting.md) |
