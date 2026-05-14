# MCP 材质与视觉操作

> **适用场景**：通过 MCP 工具操作材质/Shader/纹理/粒子特效/动画控制器 | **关联文档**：[mcp-tools.md](mcp-tools.md)（通用 MCP）、[naming-rules.md](naming-rules.md)（资源命名）、[resource-api.md](resource-api.md)（资源加载）

---

## 一、核心 API

### manage_material：材质管理

| action | 说明 | 关键参数 |
|--------|------|---------|
| `create` | 创建材质 | `materialName`, `shaderName`, `savePath` |
| `set_material_color` | 设置颜色 | `materialPath`, `colorProperty`, `r/g/b/a` |
| `set_material_shader_property` | 设置 Shader 属性 | `materialPath`, `propertyName`, `propertyType`, `value` |
| `assign_material_to_renderer` | 赋给渲染器 | `target`, `materialPath`, `materialIndex` |
| `set_renderer_color` | 快捷设置渲染器颜色 | `target`, `r/g/b/a` |
| `get_material_info` | 获取材质信息 | `materialPath` |

常用 Shader 名称：

| 管线 | Shader 名称 |
|------|------------|
| URP 不透明 | `Universal Render Pipeline/Lit` |
| URP 无光照 | `Universal Render Pipeline/Unlit` |
| 标准管线 | `Standard` |
| 精灵/2D | `Sprites/Default` |
| UI | `UI/Default` |

常用颜色属性名：

| 属性 | 适用管线 | 说明 |
|------|---------|------|
| `_BaseColor` | URP | 主颜色（URP Lit/Unlit） |
| `_Color` | 标准管线 | 主颜色（Standard Shader） |
| `_EmissionColor` | URP/标准 | 自发光颜色 |

`propertyType` 值：`float`、`int`、`color`、`vector`、`texture`

---

### manage_shader：Shader 文件

| action | 说明 | 关键参数 |
|--------|------|---------|
| `create` | 创建 Shader | `name`, `path`, `contents` |
| `delete` | 删除 Shader | `name`, `path` |

---

### manage_texture：纹理导入设置

| action | 说明 | 关键参数 |
|--------|------|---------|
| `set_import_settings` | 修改导入设置 | `path`, `maxSize`, `format`, `generateMipMaps`, `textureType` |

`textureType`：`Default`、`Sprite`、`NormalMap`、`GUI`、`Cubemap`

---

### manage_vfx：粒子与特效

#### ParticleSystem 操作

| action | 说明 | 关键参数 |
|--------|------|---------|
| `particle_create` | 创建粒子系统 | `target`, `autoAssignMaterial` |
| `particle_set_main` | 主模块 | `duration`, `looping`, `startLifetime`, `startSpeed`, `startSize`, `startColor`, `maxParticles`, `simulationSpace` |
| `particle_set_emission` | 发射模块 | `rateOverTime`, `rateOverDistance` |
| `particle_add_burst` | 爆发发射 | `time`, `count`, `cycles` |
| `particle_set_shape` | 形状模块 | `shapeType`（Sphere/Cone/Box/Mesh）, `radius`, `arc` |
| `particle_play` / `particle_stop` / `particle_clear` | 播放控制 | `target` |

#### LineRenderer 操作

| action | 说明 | 关键参数 |
|--------|------|---------|
| `line_create` | 创建线段 | `target`, `positions`（坐标数组）, `startWidth`, `endWidth` |

---

### manage_animation：动画控制器

| action | 说明 | 关键参数 |
|--------|------|---------|
| `create_controller` | 创建 AnimatorController | `controllerPath` |
| `add_parameter` | 添加参数 | `controllerPath`, `parameterName`, `parameterType`（Float/Int/Bool/Trigger）, `defaultValue` |
| `add_state` | 添加状态 | `controllerPath`, `stateName`, `clipPath`, `isDefault` |
| `add_transition` | 添加过渡 | `controllerPath`, `fromState`, `toState`, `hasExitTime`, `conditions` |
| `create_clip` | 创建动画片段 | `clipPath`, `frameRate`, `isLooping` |
| `create_blend_tree` | 创建混合树 | `controllerPath`, `stateName`, `blendType`（1D/2D）, `blendParameter`, `motions` |
| `set_parameter` | 运行时设置参数 | `target`, `parameterName`, `value` |

---

## 二、使用模式

### 材质创建完整流程

```json
// 步骤 1：创建 URP Lit 材质
{ "tool": "manage_material", "params": {
  "action": "create",
  "materialName": "EnemyMat",
  "shaderName": "Universal Render Pipeline/Lit",
  "savePath": "Assets/AssetRaw/Materials/EnemyMat.mat"
} }

// 步骤 2：设置主颜色（URP 用 _BaseColor）
{ "tool": "manage_material", "params": {
  "action": "set_material_color",
  "materialPath": "Assets/AssetRaw/Materials/EnemyMat.mat",
  "colorProperty": "_BaseColor",
  "r": 0.8, "g": 0.2, "b": 0.2, "a": 1.0
} }

// 步骤 3：设置自发光
{ "tool": "manage_material", "params": {
  "action": "set_material_shader_property",
  "materialPath": "Assets/AssetRaw/Materials/EnemyMat.mat",
  "propertyName": "_EmissionColor",
  "propertyType": "color",
  "value": { "r": 0.5, "g": 0.0, "b": 0.0, "a": 1.0 }
} }

// 步骤 4：赋给场景对象的渲染器
{ "tool": "manage_material", "params": {
  "action": "assign_material_to_renderer",
  "target": "EnemyModel",
  "materialPath": "Assets/AssetRaw/Materials/EnemyMat.mat",
  "materialIndex": 0
} }
```

---

### 粒子特效完整流程（击中特效）

```json
// 步骤 1：在已有 GameObject 上创建粒子系统
{ "tool": "manage_vfx", "params": {
  "action": "particle_create",
  "target": "HitEffect",
  "autoAssignMaterial": true
} }

// 步骤 2：设置主模块（短暂爆发效果）
{ "tool": "manage_vfx", "params": {
  "action": "particle_set_main",
  "target": "HitEffect",
  "duration": 0.5,
  "looping": false,
  "startLifetime": 0.3,
  "startSpeed": 3.0,
  "startSize": 0.2,
  "startColor": { "r": 1.0, "g": 0.6, "b": 0.1, "a": 1.0 },
  "maxParticles": 30,
  "simulationSpace": "World"
} }

// 步骤 3：设置发射模块（低速率 + 一次爆发）
{ "tool": "manage_vfx", "params": {
  "action": "particle_set_emission",
  "target": "HitEffect",
  "rateOverTime": 0
} }

// 步骤 4：添加爆发发射
{ "tool": "manage_vfx", "params": {
  "action": "particle_add_burst",
  "target": "HitEffect",
  "time": 0,
  "count": 20,
  "cycles": 1
} }

// 步骤 5：设置球形发射形状
{ "tool": "manage_vfx", "params": {
  "action": "particle_set_shape",
  "target": "HitEffect",
  "shapeType": "Sphere",
  "radius": 0.1
} }
```

---

### 动画控制器完整流程（角色移动状态机）

```json
// 步骤 1：创建控制器
{ "tool": "manage_animation", "params": {
  "action": "create_controller",
  "controllerPath": "Assets/AssetRaw/Animations/Hero.controller"
} }

// 步骤 2：添加 Speed 参数
{ "tool": "manage_animation", "params": {
  "action": "add_parameter",
  "controllerPath": "Assets/AssetRaw/Animations/Hero.controller",
  "parameterName": "Speed",
  "parameterType": "Float",
  "defaultValue": 0.0
} }

// 步骤 3：添加 Idle 状态（默认）
{ "tool": "manage_animation", "params": {
  "action": "add_state",
  "controllerPath": "Assets/AssetRaw/Animations/Hero.controller",
  "stateName": "Idle",
  "clipPath": "Assets/AssetRaw/Animations/HeroIdle.anim",
  "isDefault": true
} }

// 步骤 4：添加 Run 状态
{ "tool": "manage_animation", "params": {
  "action": "add_state",
  "controllerPath": "Assets/AssetRaw/Animations/Hero.controller",
  "stateName": "Run",
  "clipPath": "Assets/AssetRaw/Animations/HeroRun.anim",
  "isDefault": false
} }

// 步骤 5：添加 Idle→Run 过渡（Speed > 0.1）
{ "tool": "manage_animation", "params": {
  "action": "add_transition",
  "controllerPath": "Assets/AssetRaw/Animations/Hero.controller",
  "fromState": "Idle",
  "toState": "Run",
  "hasExitTime": false,
  "conditions": [{ "parameter": "Speed", "mode": "Greater", "threshold": 0.1 }]
} }

// 步骤 6：添加 Run→Idle 过渡（Speed < 0.1）
{ "tool": "manage_animation", "params": {
  "action": "add_transition",
  "controllerPath": "Assets/AssetRaw/Animations/Hero.controller",
  "fromState": "Run",
  "toState": "Idle",
  "hasExitTime": false,
  "conditions": [{ "parameter": "Speed", "mode": "Less", "threshold": 0.1 }]
} }
```

---

### 纹理导入设置（UI 图标）

```json
{ "tool": "manage_texture", "params": {
  "action": "set_import_settings",
  "path": "Assets/AssetRaw/UI/Icons/item_sword.png",
  "maxSize": 512,
  "format": "RGBA32",
  "generateMipMaps": false,
  "textureType": "Sprite"
} }
```

---

### batch_execute 批量操作（推荐）

多个视觉操作应合并为一个 `batch_execute` 调用：

```json
{ "tool": "batch_execute", "commands": [
  { "tool": "manage_material", "params": {
    "action": "create", "materialName": "HeroMat",
    "shaderName": "Universal Render Pipeline/Lit",
    "savePath": "Assets/AssetRaw/Materials/HeroMat.mat"
  } },
  { "tool": "manage_material", "params": {
    "action": "set_material_color",
    "materialPath": "Assets/AssetRaw/Materials/HeroMat.mat",
    "colorProperty": "_BaseColor",
    "r": 0.2, "g": 0.5, "b": 0.9, "a": 1.0
  } },
  { "tool": "manage_material", "params": {
    "action": "assign_material_to_renderer",
    "target": "HeroModel",
    "materialPath": "Assets/AssetRaw/Materials/HeroMat.mat",
    "materialIndex": 0
  } }
], "failFast": true }
```

---

## 三、常见错误

| 错误写法 | 正确写法 | 原因 |
|---------|---------|------|
| `colorProperty: "_Color"` 用于 URP 材质 | `colorProperty: "_BaseColor"` | `_Color` 是标准管线属性；URP Lit/Unlit 使用 `_BaseColor` |
| `shaderName: "Lit"` | `shaderName: "Universal Render Pipeline/Lit"` | Shader 名称须使用完整路径（含管线前缀） |
| `shaderName: "Standard"` 用于 URP 项目 | `shaderName: "Universal Render Pipeline/Lit"` | TEngine 项目使用 URP，Standard Shader 在 URP 下渲染异常 |
| `manage_script` 创建材质 | `manage_material` action=`create` | 材质是资产不是脚本，应使用 `manage_material` 工具 |
| `particle_set_main` 中省略 `simulationSpace` | 明确设置 `"simulationSpace": "World"` | 默认 Local 空间会导致粒子随 GameObject 移动，击中特效通常需要 World 空间 |
| `add_transition` 中 `hasExitTime: true` 用于状态响应 | `hasExitTime: false` | `hasExitTime: true` 需等动画播完才过渡，移动/战斗状态需即时响应 |
| `manage_vfx` action=`create` | `manage_vfx` action=`particle_create` | 粒子创建的正确 action 是 `particle_create`，不是 `create` |
| `propertyType: "Color"` | `propertyType: "color"` | propertyType 值小写 |

---

## 四、交叉引用

| 主题 | 文档 |
|------|------|
| 通用 MCP 操作（batch_execute/场景/脚本） | [mcp-tools.md](mcp-tools.md) |
| 资源文件命名与路径约定 | [naming-rules.md](naming-rules.md) |
| 资源加载/卸载 API（运行时） | [resource-api.md](resource-api.md) |
| UI Prefab 拼接与组件操作 | [mcp-tools.md](mcp-tools.md#ui-prefab-拼接) |
