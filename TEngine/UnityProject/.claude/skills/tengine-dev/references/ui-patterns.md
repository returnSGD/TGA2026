# UI 开发模式与模板

> **适用场景**：UIWidget 8 种创建方式（CreateWidget/CreateWidgetByPath/CreateWidgetByPrefab 等）、AdjustIconNum 列表数量管理、动态列表复用模式 | **关联文档**：[ui-lifecycle.md](ui-lifecycle.md)（生命周期）、[naming-rules.md](naming-rules.md)（节点前缀）

## 一、核心 API

### UIWidget 创建方式（8 个方法）

| # | 方法签名 | 说明 | 同步/异步 |
|---|---------|------|----------|
| 1 | `CreateWidget<T>(string goPath, bool visible = true)` | 从子节点路径创建（Prefab 中已有节点） | 同步 |
| 2 | `CreateWidget<T>(Transform parentTrans, string goPath, bool visible = true)` | 从父节点 + 子路径创建 | 同步 |
| 3 | `CreateWidget<T>(GameObject goRoot, bool visible = true)` | 从 GameObject 创建 | 同步 |
| 4 | `CreateWidgetByPath<T>(Transform parentTrans, string assetLocation, bool visible = true)` | 从资源路径创建，内部调用 `LoadGameObject` | 同步 |
| 5 | `CreateWidgetByPathAsync<T>(Transform parentTrans, string assetLocation, bool visible = true)` | 从资源路径创建，内部调用 `LoadGameObjectAsync` | 异步 |
| 6 | `CreateWidgetByPrefab<T>(GameObject goPrefab, Transform parentTrans = null, bool visible = true)` | 从预制体克隆创建 | 同步 |
| 7 | `CreateWidgetByType<T>(Transform parentTrans, bool visible = true)` | 用类型名作资源路径，内部调用 `CreateWidgetByPath` | 同步 |
| 8 | `CreateWidgetByTypeAsync<T>(Transform parentTrans, bool visible = true)` | 用类型名作资源路径，内部调用 `CreateWidgetByPathAsync` | 异步 |

> **约束**：所有泛型 `T` 必须满足 `where T : UIWidget, new()`。

### 列表管理方法（2 个方法）

```csharp
// 同步调整列表数量
void AdjustIconNum<T>(List<T> listIcon, int number, Transform parentTrans,
    GameObject prefab = null, string assetPath = "")
    where T : UIWidget, new()

// 异步调整列表数量（支持分帧创建 + 逐个回调）
void AsyncAdjustIconNum<T>(List<T> listIcon, int tarNum, Transform parentTrans,
    GameObject prefab = null, string assetPath = "",
    int maxNumPerFrame = 5, Action<T, int> updateAction = null)
    where T : UIWidget, new()
```

**AdjustIconNum 逻辑**：若 `prefab != null` 用 `CreateWidgetByPrefab`，否则用 `CreateWidgetByType`。

**AsyncAdjustIconNum 逻辑**：若 `prefab != null` 用 `CreateWidgetByPrefab`，否则用 `CreateWidgetByPathAsync`（注意：此处即使无 prefab 也走异步路径，与同步版不同）。每帧最多创建 `maxNumPerFrame` 个，每创建一个调用 `updateAction(widget, index)`。

---

## 二、使用模式

### 完整 UIWindow 示例

```csharp
[Window(UILayer.UI, "BattleMainUI", fullScreen: true)]
public class BattleMainUI : UIWindow
{
    private Button    _btnBack;
    private Text      _textHp;
    private Text      _textGold;
    private Transform _tfSkillPanel;
    private readonly List<SkillSlotWidget> _skillSlots = new();

    protected override void ScriptGenerator()
    {
        _btnBack      = FindChildComponent<Button>("m_btn_Back");
        _textHp       = FindChildComponent<Text>("m_text_Hp");
        _textGold     = FindChildComponent<Text>("m_text_Gold");
        _tfSkillPanel = FindChild("m_tf_SkillPanel");
        RegisterButtonClick(_btnBack, OnBackClicked);
    }

    protected override void RegisterEvent()
    {
        AddUIEvent<int>(IBattleEvent_Event.OnHpChanged, RefreshHp);
        AddUIEvent<int>(IBattleEvent_Event.OnGoldChanged, RefreshGold);
    }

    protected override void OnCreate()
    {
        for (int i = 0; i < 4; i++)
        {
            var slot = CreateWidget<SkillSlotWidget>($"m_tf_SkillPanel/Slot_{i}");
            _skillSlots.Add(slot);
        }
    }

    protected override void OnRefresh()
    {
        RefreshHp(PlayerData.Hp);
        RefreshGold(PlayerData.Gold);
        for (int i = 0; i < _skillSlots.Count; i++)
            _skillSlots[i].SetData(PlayerData.Skills[i]);
    }

    private void RefreshHp(int hp)    => _textHp.text   = $"HP: {hp}";
    private void RefreshGold(int gold) => _textGold.text = $"Gold: {gold}";
    private void OnBackClicked() => GameModule.UI.CloseUI<BattleMainUI>();
}
```

### UIWidget 模板

```csharp
public class ItemWidget : UIWidget
{
    private Text  _textName;
    private Image _imgIcon;

    protected override void ScriptGenerator()
    {
        _textName = FindChildComponent<Text>("m_text_Name");
        _imgIcon  = FindChildComponent<Image>("m_img_Icon");
    }

    public void SetData(ItemConfig cfg)
    {
        _textName.text = cfg.Name;
        _imgIcon.SetSprite(cfg.IconPath);  // 内置缓存池，无需手动释放
    }
}
```

### Widget 创建方法速查

```csharp
// 1. Prefab 中已有节点 — 最常用
var w1 = CreateWidget<ItemWidget>("path/to/node");

// 2. 指定父级 + 子路径
var w2 = CreateWidget<ItemWidget>(parentTrans, "goPath");

// 3. 直接用 GameObject
var w3 = CreateWidget<ItemWidget>(goRoot);

// 4. 从资源路径创建（同步，会阻塞）
var w4 = CreateWidgetByPath<ItemWidget>(parent, "Assets/ItemWidget.prefab");

// 5. 从资源路径创建（异步，推荐）
var w5 = await CreateWidgetByPathAsync<ItemWidget>(parent, "Assets/ItemWidget.prefab");

// 6. 从预制体克隆 — 列表项常用
var w6 = CreateWidgetByPrefab<ItemWidget>(prefab, parent);

// 7. 用类型名作资源路径（同步）— 资源名须与类型名一致
var w7 = CreateWidgetByType<ItemWidget>(parentTrans);

// 8. 用类型名作资源路径（异步）
var w8 = await CreateWidgetByTypeAsync<ItemWidget>(parentTrans);
```

### 列表 Widget 复用

```csharp
// 同步调整数量（prefab 优先，无 prefab 时用 CreateWidgetByType）
AdjustIconNum<ItemWidget>(listIcon, number: items.Count, parent, prefab: prefab);

// 同步调整数量（指定 assetPath 时用 CreateWidgetByPath）
AdjustIconNum<ItemWidget>(listIcon, number: items.Count, parent, assetPath: "Assets/ItemWidget.prefab");

// 异步调整数量（支持分帧创建，每帧最多 5 个）
AsyncAdjustIconNum<ItemWidget>(listIcon, tarNum: items.Count, parent,
    prefab: prefab, maxNumPerFrame: 3,
    updateAction: (widget, idx) => widget.SetData(items[idx]));

// 刷新每个 Widget（同步版用后手动刷新）
for (int i = 0; i < items.Count; i++)
    listIcon[i].SetData(items[i]);
```

### 手动绑定 API

```csharp
protected override void ScriptGenerator()
{
    var trans = FindChild("m_tf_Container");
    var btn   = FindChildComponent<Button>("m_btn_Start");
    var rect  = FindChildComponent<RectTransform>("m_rect_Panel");
    var tmp   = FindChildComponent<TextMeshProUGUI>("m_tmp_Name");
    RegisterButtonClick(btn, OnStartClicked);
}
```

---

## 三、常见错误

1. **混淆同步/异步创建**：`CreateWidgetByPath` 是同步阻塞，`CreateWidgetByPathAsync` 是异步非阻塞。在 UIWindow 的 `OnCreate` 中优先使用异步版本，避免卡顿。
2. **CreateWidgetByType 的资源命名**：该方法用 `typeof(T).Name` 作为 assetLocation，所以资源名必须与 Widget 类名完全一致，否则加载失败。
3. **AsyncAdjustIconNum 无 prefab 时走异步路径**：无 prefab 时内部调用 `CreateWidgetByPathAsync`（异步），而同步版 `AdjustIconNum` 无 prefab 时调用 `CreateWidgetByType`（同步）。两者行为不一致，混用可能导致部分 Widget 异步创建未完成就被访问。
4. **AsyncAdjustIconNum 返回 void**：该方法内部用 `.Forget()` 启动异步任务，调用方无法 await 等待完成。如需等待创建完成再操作，应手动循环使用 `CreateWidgetByPathAsync`。
5. **AdjustIconNum 的 assetPath 参数**：同步版 `AdjustIconNum` 声明了 `assetPath` 参数，但内部实现并未使用该参数（只看 `prefab` 是否为 null 来决定用 `CreateWidgetByType` 还是 `CreateWidgetByPrefab`），传 `assetPath` 无实际效果。

---

## 四、交叉引用

| 主题 | 文档 |
|------|------|
| UIWindow 生命周期、层级、属性 | [ui-lifecycle.md](ui-lifecycle.md) |
| 事件系统（AddUIEvent / GameEvent） | [event-system.md](event-system.md) |
| 节点前缀命名规范 | [naming-rules.md](naming-rules.md#ui-节点命名规范) |
| 资源加载/卸载 API | [resource-api.md](resource-api.md) |
