# 命名规范与禁止模式

> **适用场景**：C# 类型/成员命名约定、UI 节点前缀规范、禁止的代码模式（Resources.Load/Coroutine/SetSpriteAsync 等）| **关联文档**：[ui-lifecycle.md](ui-lifecycle.md)（节点前缀）、[resource-api.md](resource-api.md)（禁止模式）

## 核心 API

### C# 类型命名

| 类型 | 规范 | 示例 |
|------|------|------|
| 模块接口 | `IXxxModule` | `IResourceModule` |
| 模块实现 | `XxxModule` | `ResourceModule` |
| 事件接口 | `IXxxEvent` / `IXxxUI` + `[EventInterface]` | `ILoginUI` |
| UIWindow 子类 | `XxxUI` / `XxxWindow` | `BattleMainUI` |
| UIWidget 子类 | `XxxWidget` / `XxxItem` | `SkillSlotWidget` |
| 流程状态 | `ProcedureXxx` | `ProcedureLogin` |
| 状态机状态 | `XxxState` | `IdleState` |
| 系统类 | `XxxSystem` | `LoginSystem` |
| 配置类（Luban） | `TbXxx` / `Xxx`（行数据） | `TbItem` / `Item` |
| 内存池对象 | 实现 `IMemory` | `DamageInfo : IMemory` |

#### 字段与方法

```csharp
// 私有字段：_小驼峰（组件引用见下方 UI 前缀）
private int _currentHp;
private const int MAX_LEVEL = 100;      // 常量全大写下划线
public int CurrentHp => _currentHp;     // 公开属性大驼峰
public event Action OnDead;             // 事件大驼峰

// 异步方法：Async 后缀
public async UniTask LoadDataAsync() { }
// 事件回调：On 前缀
private void OnHpChanged(int hp) { }
```

---

### UI 节点命名规范

Prefab 节点前缀决定 `UIScriptGenerator` 自动生成的绑定类型（基于 `ScriptGeneratorSetting.asset`）：

| 前缀 | 生成类型 | 示例节点名 |
|------|---------|----------|
| `m_go_` | `GameObject` | `m_go_Effect` |
| `m_item_` | `UIWidget`（子类）| `m_item_Slot` |
| `m_tf_` | `Transform` | `m_tf_Container` |
| `m_rect_` | `RectTransform` | `m_rect_Panel` |
| `m_text_` | `Text` | `m_text_Title` |
| `m_richText_` | `RichTextItem` | `m_richText_Desc` |
| `m_btn_` | `Button` | `m_btn_Start` |
| `m_img_` | `Image` | `m_img_Icon` |
| `m_rimg_` | `RawImage` | `m_rimg_Avatar` |
| `m_scroll_` | `ScrollRect` | `m_scroll_List` |
| `m_scrollBar_` | `Scrollbar` | `m_scrollBar_Vert` |
| `m_input_` | `InputField` | `m_input_Name` |
| `m_grid_` | `GridLayoutGroup` | `m_grid_Items` |
| `m_hlay_` | `HorizontalLayoutGroup` | `m_hlay_Tabs` |
| `m_vlay_` | `VerticalLayoutGroup` | `m_vlay_List` |
| `m_slider_` | `Slider` | `m_slider_Volume` |
| `m_toggle_` | `Toggle` | `m_toggle_Sound` |
| `m_group_` | `ToggleGroup` | `m_group_Tab` |
| `m_curve_` | `AnimationCurve` | `m_curve_Anim` |
| `m_canvasGroup_` | `CanvasGroup` | `m_canvasGroup_Fade` |
| `m_tmp_` | `TextMeshProUGUI` | `m_tmp_Name` |
| `m_tmpInput_` | `TMP_InputField` | `m_tmpInput_Search` |
| `m_tmpDropdown_` | `TMP_Dropdown` | `m_tmpDropdown_Lang` |
| `m_canvas_` | `Canvas` | `m_canvas_Overlay` |
| `m_dropdown_` | `Dropdown` | `m_dropdown_Select` |

**注意**：
- 前缀匹配规则来自 `ScriptGeneratorSetting.asset` 中的 `uiElementRegex` 字段，匹配时不带尾部下划线（如 regex 是 `m_tmpInput`，节点名 `m_tmpInput_XXX` 可匹配）
- `m_scrollBar_` 必须写在 `m_scroll_` 之前，否则 `m_scrollBar_X` 会被 `m_scroll` 先匹配（setting 中确实 scrollBar 在 scroll 之前）
- `m_tmpInput_` 和 `m_tmpDropdown_` 必须写在 `m_tmp_` 之前，同理（setting 中 tmpInput/tmpDropdown 在 tmp 之前）
- `m_richText_` 必须写在 `m_text_` 之前（setting 中 richText 在 text 之前）
- 不需要绑定的节点无需加前缀，生成器会忽略

---

## 使用模式

### 异步编程规范

```csharp
// ✅ UniTask 替代 Task，UniTaskVoid 替代 void async
public async UniTask<int> GetDataAsync() { }
public async UniTaskVoid StartBattleAsync() { }  // 调用方加 .Forget()

// ✅ CancellationToken 防止销毁后回调
private CancellationTokenSource _cts = new();
protected override void OnDestroy() { _cts.Cancel(); _cts.Dispose(); }

// ✅ 并发加载
var (a, b, c) = await UniTask.WhenAll(LoadA(), LoadB(), LoadC());
```

---

## 常见错误

| 错误 | 原因 | 修复 |
|------|------|------|
| 节点 `m_tInput_Search` 不生成 TMP_InputField | 旧文档前缀 `m_tInput_` 已过时 | 正确前缀为 `m_tmpInput_` |
| `m_tmp_Search` 被识别为 TMP_InputField | `m_tmpInput` regex 先于 `m_tmp` 匹配 | 节点名不含 `Input` 时不会被误匹配 |
| 前缀匹配顺序错误 | regex 按序匹配，长前缀需先声明 | scrollBar 在 scroll 前、tmpInput/tmpDropdown 在 tmp 前 |
| 缺少 TMP_Dropdown 绑定 | 旧文档未记录此前缀 | 使用 `m_tmpDropdown_` 前缀 |
| 缺少 Canvas/Dropdown 绑定 | 旧文档未记录此前缀 | 使用 `m_canvas_` / `m_dropdown_` 前缀 |

### 禁止的异步模式

```csharp
// ❌ Task → 用 UniTask
// ❌ Coroutine → 用 async/await
// ❌ Update 中 await → 用 Timer
// ❌ 忽略 UniTask 返回值 → 加 .Forget() 或 await
```

### 禁止的代码模式

```csharp
// ❌ Resources.Load → GameModule.Resource
// ❌ Instantiate(prefab) → LoadGameObjectAsync
// ❌ FindObjectOfType → GameModule 或事件
// ❌ Update 中 new 对象 → MemoryPool
// ❌ 跨模块强引用 → GameEvent
// ❌ 外部访问 UI 私有组件 → 事件或公共方法
// ❌ 静态持有 Asset 引用 → 内存泄漏
// ❌ 忽略 async 返回值 → await 或 .Forget()
```

---

## 交叉引用

- 架构总览见 [architecture.md](architecture.md)
- UI 生命周期见 [ui-lifecycle.md](ui-lifecycle.md)
- UI 进阶模式见 [ui-patterns.md](ui-patterns.md)
