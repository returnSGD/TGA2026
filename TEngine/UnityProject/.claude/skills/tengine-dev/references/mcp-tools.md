# MCP 工具指南

> **适用场景**：MCP 场景管理（场景加载/对象创建）、GameObject 操作、UI Prefab 创建、脚本生成、编辑器自动化、测试工具 | **关联文档**：[mcp-visual.md](mcp-visual.md)（材质/Shader/VFX/动画）、[naming-rules.md](naming-rules.md)（命名约定）

## 核心原则：batch_execute 优先

批量操作比单次调用快 10~100 倍，多对象任务必须用 `batch_execute`：

```json
{
  "tool": "batch_execute",
  "commands": [
    { "tool": "manage_gameobject", "params": { "action": "create", "name": "Root" } },
    { "tool": "manage_gameobject", "params": { "action": "create", "name": "Child", "parent": "Root" } },
    { "tool": "manage_components", "params": { "action": "add", "target": "Child", "componentType": "Button" } }
  ],
  "failFast": true
}
```

- 默认每批最多 25 条命令（可配置，最大 100）
- `failFast: true` 遇到第一个错误即停止

## 目标定位方式（target）

| 方式 | 示例值 | 说明 |
|------|--------|------|
| 名称 | `"Canvas"` | 场景中第一个匹配 |
| 层级路径 | `"UIRoot/Canvas/Panel"` | 含 `/` 时按路径查找 |
| InstanceID | `12345` | 最可靠，不受重名影响 |
| Tag | `searchMethod: "by_tag"` | 查找指定 Tag 对象 |

`searchMethod`：`by_name`（默认）、`by_path`、`by_id`、`by_tag`

---

## 场景与 GameObject

### manage_scene

| action | 说明 | 关键参数 |
|--------|------|---------|
| `get_active` | 当前场景信息 | — |
| `get_hierarchy` | 场景层级（分页） | `parent`, `pageSize`, `cursor`, `maxDepth` |
| `save` | 保存当前场景 | — |
| `load` | 加载场景 | `name`, `path`（需先 save） |
| `create` | 创建新场景 | `name`, `path` |
| `screenshot` | 截图到 Assets/Screenshots/ | `fileName`, `superSize` |
| `get_build_settings` | Build Settings 场景列表 | — |

`get_hierarchy` 返回含 `next_cursor`，`truncated=true` 时需翻页。

### manage_gameobject

| action | 说明 | 关键参数 |
|--------|------|---------|
| `create` | 创建 GO | `name`, `parent`, `position`, `componentType`, `primitiveType` |
| `modify` | 修改属性 | `target`, `position`, `rotation`, `scale`, `setActive`, `name`, `parent` |
| `delete` | 删除 GO | `target` |
| `duplicate` | 复制 GO | `target`, `name`, `position` |
| `move_relative` | 相对移动 | `target`, `deltaPosition`, `space`（World/Self） |

`primitiveType`：Cube/Sphere/Plane/Cylinder/Capsule/Quad

### find_gameobjects

| searchMethod | 说明 | 常用参数 |
|-------------|------|---------|
| `by_name` | 按名称 | `query`, `includeInactive`, `maxResults` |
| `by_path` | 按层级路径 | `query`（如 `"Root/Parent/Child"`） |
| `by_tag` | 按 Tag | `query`, `maxResults` |
| `by_layer` | 按 Layer | `query` |

返回：`name`、`instanceID`、`path`、`componentTypes`、`activeSelf`

### manage_components

| action | 说明 | 关键参数 |
|--------|------|---------|
| `add` | 添加组件 | `target`, `componentType`, `properties` |
| `remove` | 删除组件 | `target`, `componentType` |
| `set_property` | 设置组件属性 | `target`, `componentType`, `properties` |

常用组件类型：`Rigidbody`/`Rigidbody2D`、`BoxCollider`/`SphereCollider`、`MeshRenderer`/`SpriteRenderer`、`Light`、`Camera`、`AudioSource`、`Animator`、`NavMeshAgent`、`ParticleSystem`、`Canvas`/`CanvasScaler`/`GraphicRaycaster`、`VerticalLayoutGroup`/`HorizontalLayoutGroup`

### TEngine 场景约定

| 约定 | 说明 |
|------|------|
| **UIRoot** | 场景必须存在，`UIModule.OnInit()` 自动查找 |
| **场景路径** | `Assets/Scenes/` 或 `Assets/AssetRaw/Scenes/` |
| **层级** | GameRoot → Logic / UI / Effect |
| **禁止场景中直接放 UI** | UI 通过 `UIModule.ShowUIAsync` 动态加载 |

---

## UI Prefab 拼接

### 前缀与 MCP 创建工具对照

完整前缀→C#类型绑定见 [naming-rules.md](naming-rules.md#ui-节点命名规范)。

| 前缀 | MCP 创建方式 |
|------|------------|
| `m_btn_` | `manage_ui` action=`create_button` |
| `m_img_` | `manage_ui` action=`create_image` |
| `m_text_` | `manage_ui` action=`create_text` |
| `m_tmp_` | `manage_ui` action=`create_text`（自动检测 TMP） |
| `m_slider_` | `manage_ui` action=`create_slider` |
| `m_toggle_` | `manage_ui` action=`create_toggle` |
| `m_input_` | `manage_ui` action=`create_inputfield` |
| `m_go_/m_tf_/m_rect_` | `manage_gameobject` action=`create` |
| `m_rimg_/m_scroll_/m_scrollBar_` | `manage_gameobject` + `manage_components` add 对应类型 |
| `m_grid_/m_hlay_/m_vlay_/m_canvasGroup_` | `manage_gameobject` + `manage_components` add 对应类型 |
| `m_item_` | `manage_gameobject` + 挂对应 Widget 脚本 |

### Prefab 结构要求

UIModule 加载时强制检查根节点必须有 Canvas：

```
XxxUI.prefab（根节点）
├── [Canvas] ← 必须
├── [CanvasScaler] ← 强烈建议
├── [GraphicRaycaster] ← 交互必须
└── 子节点（m_btn_/m_tmp_/m_tf_/...）
```

存放：`Assets/AssetRaw/UI/Prefabs/<PrefabName>.prefab`

### Canvas 适配

| 参数 | 推荐值 |
|------|-------|
| UI Scale Mode | Scale With Screen Size |
| Reference Resolution | 1920 × 1080 |
| Screen Match Mode | Match Width Or Height |
| Match | 0.5 |

锚点规则：全屏→四角拉伸 | 弹窗→中心锚点+固定尺寸 | HUD→锚定对应边

### 标准工作流

```
1. batch_execute 创建根节点（Canvas+Scaler+Raycaster）+ 所有 UI 子节点
2. manage_prefabs action=create_from_gameobject → 保存为 Prefab
3. manage_gameobject action=delete → 清理场景临时 GO
```

### manage_ui 速查

通用参数：`name`、`parent`、`x`/`y`、`width`/`height`

| action | 特有参数 |
|--------|---------|
| `create_canvas` | `renderMode` |
| `create_panel` | `r/g/b/a` |
| `create_button` | `text` |
| `create_text` | `text`, `fontSize`, `r/g/b/a` |
| `create_image` | `spritePath`, `r/g/b/a` |
| `create_inputfield` | `placeholder` |
| `create_slider` | `minValue`, `maxValue`, `value` |
| `create_toggle` | `label`, `isOn` |
| `ui_set_text` / `ui_set_anchor` | `text` / `preset`（TopLeft/MiddleCenter/StretchAll） |
| `ui_layout_children` | `layout`, `spacing` |

### 骨架模板

#### 全屏窗口
```json
{ "tool": "batch_execute", "commands": [
  { "tool": "manage_gameobject", "params": { "action": "create", "name": "FSUI", "componentType": "Canvas" } },
  { "tool": "manage_components", "params": { "action": "add", "target": "FSUI", "componentType": "CanvasScaler" } },
  { "tool": "manage_components", "params": { "action": "add", "target": "FSUI", "componentType": "GraphicRaycaster" } },
  { "tool": "manage_ui", "params": { "action": "create_panel", "name": "m_tf_Bg", "parent": "FSUI", "width": 1920, "height": 1080, "r": 0, "g": 0, "b": 0, "a": 0.8 } },
  { "tool": "manage_ui", "params": { "action": "create_text", "name": "m_tmp_Title", "parent": "FSUI", "text": "标题", "fontSize": 48, "y": 400 } },
  { "tool": "manage_ui", "params": { "action": "create_button", "name": "m_btn_Close", "parent": "FSUI", "text": "×", "x": 600, "y": 400 } },
  { "tool": "manage_gameobject", "params": { "action": "create", "name": "m_tf_Content", "parent": "FSUI" } }
], "failFast": true }
```

#### 弹窗
```json
{ "tool": "batch_execute", "commands": [
  { "tool": "manage_gameobject", "params": { "action": "create", "name": "PopUI", "componentType": "Canvas" } },
  { "tool": "manage_components", "params": { "action": "add", "target": "PopUI", "componentType": "CanvasScaler" } },
  { "tool": "manage_components", "params": { "action": "add", "target": "PopUI", "componentType": "GraphicRaycaster" } },
  { "tool": "manage_ui", "params": { "action": "create_panel", "name": "m_img_Mask", "parent": "PopUI", "width": 1920, "height": 1080, "a": 0.5 } },
  { "tool": "manage_ui", "params": { "action": "create_panel", "name": "m_tf_Win", "parent": "PopUI", "width": 600, "height": 400, "r": 0.2, "g": 0.2, "b": 0.2 } },
  { "tool": "manage_ui", "params": { "action": "create_text", "name": "m_tmp_Title", "parent": "m_tf_Win", "text": "提示", "fontSize": 32, "y": 150 } },
  { "tool": "manage_ui", "params": { "action": "create_button", "name": "m_btn_OK", "parent": "m_tf_Win", "text": "确认", "x": 100, "y": -150 } },
  { "tool": "manage_ui", "params": { "action": "create_button", "name": "m_btn_Cancel", "parent": "m_tf_Win", "text": "取消", "x": -100, "y": -150 } }
], "failFast": true }
```

#### 列表（带 ScrollView）
```json
{ "tool": "batch_execute", "commands": [
  { "tool": "manage_gameobject", "params": { "action": "create", "name": "ListUI", "componentType": "Canvas" } },
  { "tool": "manage_components", "params": { "action": "add", "target": "ListUI", "componentType": "CanvasScaler" } },
  { "tool": "manage_components", "params": { "action": "add", "target": "ListUI", "componentType": "GraphicRaycaster" } },
  { "tool": "manage_gameobject", "params": { "action": "create", "name": "m_scroll_List", "parent": "ListUI" } },
  { "tool": "manage_components", "params": { "action": "add", "target": "m_scroll_List", "componentType": "ScrollRect" } },
  { "tool": "manage_gameobject", "params": { "action": "create", "name": "m_tf_Content", "parent": "m_scroll_List" } },
  { "tool": "manage_components", "params": { "action": "add", "target": "m_tf_Content", "componentType": "VerticalLayoutGroup" } }
], "failFast": true }
```

### Prefab headless 编辑

不打开 Prefab Stage，直接修改 .prefab 文件：
```json
{ "tool": "manage_prefabs", "params": {
  "action": "modify_contents",
  "prefabPath": "Assets/AssetRaw/UI/Prefabs/XxxUI.prefab",
  "target": "m_btn_Back",
  "componentProperties": { "RectTransform": { "localPosition": { "x": -400, "y": 250, "z": 0 } } }
} }
```

查看层级：`manage_prefabs` action=`get_hierarchy`

---

## 脚本与资源管理

### manage_script：C# 脚本管理

| action | 说明 | 关键参数 |
|--------|------|---------|
| `create` | 创建脚本 | `name`, `path`, `contents`, `namespace` |
| `delete` | 删除脚本 | `name`, `path` |
| `get_sha` | 获取 SHA256 | `name`, `path`（编辑前必须先获取） |
| `validate` | 验证语法 | `name`, `path`, `level`（basic/standard/comprehensive/strict） |

```json
{ "tool": "manage_script", "params": {
  "action": "create", "name": "BattleMainUI",
  "path": "Assets/GameScripts/HotFix/GameLogic/UI/Battle",
  "contents": "using TEngine;\nnamespace GameLogic\n{\n    [Window(UILayer.UI, \"BattleMainUI\")]\n    public class BattleMainUI : UIWindow { }\n}",
  "namespace": "GameLogic"
} }
```

UIWindow/UIWidget 骨架模板见 [ui-patterns.md](ui-patterns.md)。

### apply_text_edits：精确文本编辑

**推荐流程：读取 → get_sha → 精确编辑**，避免全量覆写风险。

```json
{ "tool": "apply_text_edits", "params": {
  "name": "BattleMainUI",
  "path": "Assets/GameScripts/HotFix/GameLogic/UI/Battle",
  "precondition_sha256": "<上一步的sha256>",
  "edits": [
    { "startLine": 15, "startCol": 1, "endLine": 18, "endCol": 1,
      "newText": "        protected override void OnRefresh()\n        {\n            RefreshHp(PlayerData.Hp);\n        }\n" }
  ],
  "options": { "refresh": "debounced", "validate": "standard" }
} }
```

规则：`precondition_sha256` 必须匹配当前文件 | 行列从 1 开始 | 多编辑区域不能重叠 | `refresh: "debounced"` 延迟合并编译

| 错误码 | 处理 |
|--------|------|
| `precondition_required` | 先调用 `get_sha` |
| `stale_file` | 重新获取 SHA |
| `overlap` | 按行号降序排列编辑项 |

### manage_asset：资源文件管理

| action | 说明 | 关键参数 |
|--------|------|---------|
| `search` | 搜索资源 | `query`, `type`（Prefab/Texture2D/AudioClip/...）, `path` |
| `get_info` | 获取资源信息 | `path`（返回类型、GUID、大小、依赖项） |
| `move` | 移动资源 | `path`, `newPath` |
| `rename` | 重命名 | `path`, `newName` |
| `duplicate` | 复制资源 | `path`, `newPath` |
| `delete` | 删除资源 | `path` |
| `create_folder` | 创建文件夹 | `path` |

刷新：`refresh_unity`（manage_script 操作后自动刷新，无需手动）

### manage_scriptable_object：SO 读写

| action | 说明 | 关键参数 |
|--------|------|---------|
| `read` | 读取字段 | `path` |
| `write` | 修改字段 | `path`, `properties` |
| `create` | 创建 SO 实例 | `typeName`, `path` |

### TEngine 脚本路径约定

| 类型 | 路径 |
|------|------|
| UIWindow/Widget | `Assets/GameScripts/HotFix/GameLogic/UI/<模块名>/` |
| 生成代码（绑定） | `Assets/GameScripts/HotFix/GameLogic/UI/Gen/` |
| 模块代码 | `Assets/GameScripts/HotFix/GameLogic/Module/<ModuleName>/` |
| 事件接口 | `Assets/GameScripts/HotFix/GameLogic/Event/` |
| GameProto（Luban） | `Assets/GameScripts/HotFix/GameProto/`（自动生成，勿手改） |

---

## 编辑器控制与调试

### manage_editor

| action | 说明 |
|--------|------|
| `play` | 进入 Play Mode（`waitForCompletion: true` 等编译后进入） |
| `pause` | 暂停/恢复 |
| `stop` | 退出 Play Mode |

### Tag/Layer 管理

| action | 说明 | 参数 |
|--------|------|------|
| `add_tag` / `remove_tag` / `list_tags` | Tag CRUD | `tagName` |
| `add_layer` / `remove_layer` / `list_layers` | Layer CRUD | `layerName`（自动选 8~31 空闲槽） |

### execute_menu_item：执行菜单命令

TEngine 常用菜单：

| 操作 | menuItem 路径 |
|------|-------------|
| 刷新资源 | `Assets/Refresh` |
| 保存项目 | `File/Save Project` |
| 清理缓存 | `Tools/YooAsset/Clear Build Cache` |
| 生成 UI 脚本 | `Tools/UIScriptGenerator/Generate Selected` |
| Luban 配置生成 | `Tools/Luban/Generate` |
| HybridCLR 生成 | `HybridCLR/Generate/All` |

### run_tests / get_test_job：自动化测试

```json
{ "tool": "run_tests", "params": { "mode": "EditMode", "filter": "GameLogic.Tests" } }
```

返回 `job_id`，后台运行。每 3 秒轮询：

```json
{ "tool": "get_test_job", "params": { "job_id": "<job_id>" } }
```

`status`：`running` → `completed` / `failed`

清理卡住任务：`run_tests` params=`{ "clear_stuck": true }`

### read_console：控制台日志

```json
{ "tool": "read_console", "params": { "count": 30, "logLevel": "Error" } }
```

`logLevel`：`All`、`Log`、`Warning`、`Error`、`Exception`

支持 `filter` 关键词过滤：`{ "count": 20, "filter": "BattleMainUI", "logLevel": "All" }`

### refresh_unity

`manage_script` 操作后自动刷新，通常无需手动。

### 调试工作流

#### 场景运行调试

```
1. manage_scene action=save 保存场景
2. manage_editor action=play 进入运行
3. read_console logLevel=Error count=20 检查错误
4. manage_editor action=stop 退出
5. apply_text_edits 修复 → 等编译 → 重新运行
```

#### 编译错误排查

```
1. 修改脚本后等待编译
2. read_console logLevel=Error count=30
3. 根据错误行号 apply_text_edits 修复
4. read_console 确认无新错误
```

#### TEngine 热更重新生成

```json
{ "tool": "batch_execute", "commands": [
  { "tool": "execute_menu_item", "params": { "menuItem": "HybridCLR/Generate/All" } },
  { "tool": "refresh_unity", "params": {} }
], "failFast": true }
```

---

## 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `Target not found` | 名称不存在 | 先 `find_gameobjects` 确认 |
| `Scene has unsaved changes` | 未保存即切换场景 | 先 `manage_scene` save |
| `Prefab asset requires manage_prefabs` | 对 .prefab 用错工具 | 改用 `manage_prefabs` |
| `precondition_required` | 缺少 SHA | 先 `get_sha` |
| `stale_file` | 文件已被修改 | 重新 `get_sha` |
| `batch too large` | 单批超限 | 拆分多个 batch_execute |

视觉操作（材质/Shader/纹理/粒子/动画）见 [mcp-visual.md](mcp-visual.md)。
