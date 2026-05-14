# UI系统

<cite>
**本文引用的文件**
- [UIWindow.cs](file://Assets/GameScripts/HotFix/GameLogic/Module/UIModule/UIWindow.cs)
- [UIBase.cs](file://Assets/Launcher/Scripts/UIBase.cs)
- [LoginUI.cs](file://Assets/GameScripts/HotFix/GameLogic/UI/LoginUI/LoginUI.cs)
- [BattleMainUI.cs](file://Assets/GameScripts/HotFix/GameLogic/UI/BattleMainUI/BattleMainUI.cs)
- [HtmlToUGUIBaker.cs](file://Assets/HtmlToUGUI/Editor/HtmlToUGUIBaker.cs)
- [HtmlToUGUIConfig.cs](file://Assets/HtmlToUGUI/HtmlToUGUIConfig.cs)
- [HtmlToUGUIConfig.asset](file://Assets/HtmlToUGUI/HtmlToUGUIConfig.asset)
- [test.json](file://Assets/HtmlToUGUI/UIjson/test.json)
- [HTML界面案例.html](file://Assets/HtmlToUGUI/HTML/HTML界面案例.html)
- [Constant.cs](file://Assets/TEngine/Runtime/Core/Constant/Constant.cs)
- [GameEventMgr.cs](file://Assets/TEngine/Runtime/Core/GameEvent/GameEventMgr.cs)
</cite>

## 目录
1. [简介](#简介)
2. [项目结构](#项目结构)
3. [核心组件](#核心组件)
4. [架构总览](#架构总览)
5. [详细组件分析](#详细组件分析)
6. [依赖关系分析](#依赖关系分析)
7. [性能考量](#性能考量)
8. [故障排查指南](#故障排查指南)
9. [结论](#结论)
10. [附录](#附录)

## 简介
本文件面向TEngine UI系统，系统性梳理UI框架的整体架构与设计理念，覆盖UI组件绑定、生命周期管理、事件处理、UI生成器工具（HTML转UGUI）、窗口管理机制（显示/隐藏、层级、遮罩）以及最佳实践与扩展指南。文档以仓库中的实际代码为依据，配合图示帮助读者快速理解并高效使用该UI体系。

## 项目结构
TEngine UI系统由“运行时UI基类”、“热更新UI窗口”、“UI模块管理”、“HTML到UGUI烘焙工具”、“UI资源与脚本生成”等部分组成。核心文件分布如下：
- 运行时与热更新层：UIBase、UIWindow、具体UI窗口（如LoginUI、BattleMainUI）
- UI模块与管理：UIModule（通过UIWindow参与管理）
- UI生成器：HtmlToUGUIBaker、HtmlToUGUIConfig及配套JSON/HTML样例
- 事件系统：GameEventMgr（UI事件订阅）
- 常量与设置：Constant（如UI音效相关设置）

```mermaid
graph TB
subgraph "运行时与热更新"
UIBase["UIBase<br/>热更UI基类"]
UIWindow["UIWindow<br/>窗口抽象"]
LoginUI["LoginUI<br/>示例窗口"]
BattleUI["BattleMainUI<br/>示例窗口"]
end
subgraph "UI模块"
UIModule["UIModule<br/>窗口栈与更新循环"]
end
subgraph "UI生成器"
HtmlBaker["HtmlToUGUIBaker<br/>编辑器烘焙器"]
HtmlCfg["HtmlToUGUIConfig<br/>配置(ScriptableObject)"]
JsonData["test.json<br/>UI树JSON"]
HtmlSrc["HTML界面案例.html<br/>HTML原型"]
end
subgraph "事件与设置"
GameEvent["GameEventMgr<br/>事件管理"]
Const["Constant<br/>常量设置"]
end
UIBase --> UIWindow
UIWindow --> UIModule
LoginUI --> UIWindow
BattleUI --> UIWindow
HtmlBaker --> HtmlCfg
HtmlBaker --> JsonData
HtmlBaker --> HtmlSrc
GameEvent --> UIModule
Const --> UIModule
```

**图表来源**
- [UIBase.cs:1-94](file://Assets/Launcher/Scripts/UIBase.cs#L1-L94)
- [UIWindow.cs:1-524](file://Assets/GameScripts/HotFix/GameLogic/Module/UIModule/UIWindow.cs#L1-L524)
- [LoginUI.cs:1-14](file://Assets/GameScripts/HotFix/GameLogic/UI/LoginUI/LoginUI.cs#L1-L14)
- [BattleMainUI.cs:1-30](file://Assets/GameScripts/HotFix/GameLogic/UI/BattleMainUI/BattleMainUI.cs#L1-L30)
- [HtmlToUGUIBaker.cs:1-836](file://Assets/HtmlToUGUI/Editor/HtmlToUGUIBaker.cs#L1-L836)
- [HtmlToUGUIConfig.cs:1-35](file://Assets/HtmlToUGUI/HtmlToUGUIConfig.cs#L1-L35)
- [test.json:1-1018](file://Assets/HtmlToUGUI/UIjson/test.json#L1-L1018)
- [HTML界面案例.html:1-146](file://Assets/HtmlToUGUI/HTML/HTML界面案例.html#L1-L146)
- [GameEventMgr.cs:48-94](file://Assets/TEngine/Runtime/Core/GameEvent/GameEventMgr.cs#L48-L94)
- [Constant.cs:1-21](file://Assets/TEngine/Runtime/Core/Constant/Constant.cs#L1-L21)

**章节来源**
- [UIBase.cs:1-94](file://Assets/Launcher/Scripts/UIBase.cs#L1-L94)
- [UIWindow.cs:1-524](file://Assets/GameScripts/HotFix/GameLogic/Module/UIModule/UIWindow.cs#L1-L524)
- [HtmlToUGUIBaker.cs:1-836](file://Assets/HtmlToUGUI/Editor/HtmlToUGUIBaker.cs#L1-L836)

## 核心组件
- UIBase：热更UI基类，提供通用生命周期入口、显示/隐藏、查找子节点/组件、参数传递等能力。
- UIWindow：UI窗口抽象，继承自UIBase，负责窗口生命周期（加载、创建、刷新、更新、销毁）、层级排序、可见性与交互性、安全区域适配、定时关闭等。
- 具体UI窗口：如LoginUI、BattleMainUI，通过特性标注窗口层级与定位，内部通过脚本生成器方法完成组件绑定。
- HtmlToUGUIBaker：编辑器工具，将JSON或HTML原型烘焙为UGUI层级结构，支持多分辨率与全控件映射。
- HtmlToUGUIConfig：配置ScriptableObject，统一管理分辨率预设与DSL模板。
- GameEventMgr：事件系统封装，便于UI订阅与派发事件。
- Constant：常量设置，包含UI相关设置键名（如UI音效开关与音量）。

**章节来源**
- [UIBase.cs:1-94](file://Assets/Launcher/Scripts/UIBase.cs#L1-L94)
- [UIWindow.cs:1-524](file://Assets/GameScripts/HotFix/GameLogic/Module/UIModule/UIWindow.cs#L1-L524)
- [LoginUI.cs:1-14](file://Assets/GameScripts/HotFix/GameLogic/UI/LoginUI/LoginUI.cs#L1-L14)
- [BattleMainUI.cs:1-30](file://Assets/GameScripts/HotFix/GameLogic/UI/BattleMainUI/BattleMainUI.cs#L1-L30)
- [HtmlToUGUIBaker.cs:1-836](file://Assets/HtmlToUGUI/Editor/HtmlToUGUIBaker.cs#L1-L836)
- [HtmlToUGUIConfig.cs:1-35](file://Assets/HtmlToUGUI/HtmlToUGUIConfig.cs#L1-L35)
- [GameEventMgr.cs:48-94](file://Assets/TEngine/Runtime/Core/GameEvent/GameEventMgr.cs#L48-L94)
- [Constant.cs:1-21](file://Assets/TEngine/Runtime/Core/Constant/Constant.cs#L1-L21)

## 架构总览
UI系统采用“热更新UI基类 + 抽象窗口 + 编辑器烘焙工具”的分层设计：
- 热更新层：UIBase提供通用能力；UIWindow负责窗口生命周期与渲染层管理。
- 管理层：UIModule维护窗口栈、更新循环与层级排序。
- 设计工具层：HtmlToUGUIBaker将设计稿（HTML/JSON）转换为UGUI树，提升UI搭建效率。
- 事件与设置：GameEventMgr统一事件订阅；Constant提供设置键名。

```mermaid
classDiagram
class UIBase {
+GameObject gameObject
+RectTransform rectTransform
+bool FullScreen
+OnInit(param)
+Show()
+Hide()
+Close()
+FindChild(path)
+FindChildComponent<T>(path)
+CallScriptGenerator()
}
class UIWindow {
+string WindowName
+int WindowLayer
+bool FullScreen
+int Depth
+bool Visible
+bool Interactable
+InternalLoad(...)
+InternalCreate()
+InternalRefresh()
+InternalUpdate()
+InternalDestroy(isShutDown)
+SetUIFit(...)
+SetUINotFit(...)
}
class HtmlToUGUIBaker {
+ExecuteBake()
+ConfigureCanvasScaler(canvas)
+CreateUINode(node,parent,x,y)
+ApplyComponentByType(go,nodeData)
}
class HtmlToUGUIConfig {
+List<UIResolutionConfig> supportedResolutions
+TextAsset dslTemplateAsset
}
class GameEventMgr {
+AddEvent(eventType, handler)
+AddEvent<T>(eventType, handler)
}
UIBase <|-- UIWindow
HtmlToUGUIBaker --> HtmlToUGUIConfig : "使用"
```

**图表来源**
- [UIBase.cs:1-94](file://Assets/Launcher/Scripts/UIBase.cs#L1-L94)
- [UIWindow.cs:1-524](file://Assets/GameScripts/HotFix/GameLogic/Module/UIModule/UIWindow.cs#L1-L524)
- [HtmlToUGUIBaker.cs:1-836](file://Assets/HtmlToUGUI/Editor/HtmlToUGUIBaker.cs#L1-L836)
- [HtmlToUGUIConfig.cs:1-35](file://Assets/HtmlToUGUI/HtmlToUGUIConfig.cs#L1-L35)
- [GameEventMgr.cs:48-94](file://Assets/TEngine/Runtime/Core/GameEvent/GameEventMgr.cs#L48-L94)

## 详细组件分析

### UIBase：热更UI基类
- 职责：提供UI通用能力，如显示/隐藏、参数初始化、子节点/组件查找、脚本生成器调用。
- 关键点：
  - 提供FindChild/FindChildComponent系列方法，支持按路径查找与类型化组件获取。
  - 提供CallScriptGenerator与虚函数ScriptGenerator，用于脚本生成器绑定。
  - Show/Hide/Close分别控制激活状态与关闭流程。

**章节来源**
- [UIBase.cs:1-94](file://Assets/Launcher/Scripts/UIBase.cs#L1-L94)

### UIWindow：窗口抽象与生命周期
- 生命周期阶段：
  - 加载：InternalLoad根据资源来源异步或同步加载实例，完成后Handle_Completed并触发准备回调。
  - 创建：InternalCreate执行注入、脚本生成器、成员绑定、事件注册与OnCreate回调。
  - 刷新：InternalRefresh调用OnRefresh。
  - 更新：InternalUpdate遍历子UIWidget并调用OnUpdate，返回是否需要持续更新。
  - 销毁：InternalDestroy注销事件、销毁子UIWidget、清理回调与面板。
- 层级与可见性：
  - Depth属性通过Canvas.sortingOrder实现层级排序，并同步子Canvas。
  - Visible属性控制显示/隐藏与Raycaster交互性，同时设置图层。
- 安全区域适配：SetUIFit/SetUINotFit支持刘海屏适配与局部不参与适配。
- 定时关闭：HideTimeToClose与隐藏定时器配合，隐藏一段时间后关闭。

```mermaid
sequenceDiagram
participant Caller as "调用方"
participant Window as "UIWindow"
participant Loader as "UIModule.Resource"
participant Panel as "GameObject实例"
Caller->>Window : InternalLoad(location, callback, isAsync, userDatas)
alt 异步
Window->>Loader : LoadGameObjectAsync(location, parent)
Loader-->>Window : Panel
else 同步
Window->>Loader : LoadGameObject(location, parent)
Loader-->>Window : Panel
end
Window->>Window : Handle_Completed(panel)
Window-->>Caller : 准备回调(已准备)
Caller->>Window : InternalCreate()
Window->>Window : 注入/脚本生成/绑定/注册事件/OnCreate
Caller->>Window : InternalUpdate()
Window-->>Caller : 返回是否继续更新
Caller->>Window : InternalDestroy()
Window->>Window : 注销事件/销毁子UIWidget/OnDestroy/销毁面板
```

**图表来源**
- [UIWindow.cs:314-524](file://Assets/GameScripts/HotFix/GameLogic/Module/UIModule/UIWindow.cs#L314-L524)

**章节来源**
- [UIWindow.cs:1-524](file://Assets/GameScripts/HotFix/GameLogic/Module/UIModule/UIWindow.cs#L1-L524)

### 具体UI窗口：LoginUI 与 BattleMainUI
- LoginUI：通过Window特性标注层级，作为最小示例窗口。
- BattleMainUI：展示脚本生成器绑定流程，包含容器与多个子节点的绑定方法，演示了如何在ScriptGenerator中完成组件缓存与事件声明区域。

**章节来源**
- [LoginUI.cs:1-14](file://Assets/GameScripts/HotFix/GameLogic/UI/LoginUI/LoginUI.cs#L1-L14)
- [BattleMainUI.cs:1-30](file://Assets/GameScripts/HotFix/GameLogic/UI/BattleMainUI/BattleMainUI.cs#L1-L30)

### HTML到UGUI生成器：HtmlToUGUIBaker
- 功能概述：
  - 支持文件模式与字符串模式两种输入。
  - 通过配置ScriptableObject管理多分辨率与DSL模板。
  - 将JSON或HTML原型解析为UGUI树，自动挂载Image、Text/TextMeshProUGUI、Button、InputField/TMP_InputField、ScrollRect、Toggle、Slider、Dropdown等组件。
  - 自动设置CanvasScaler与锚点、轴心、尺寸与位置。
- 关键流程：
  - 执行烘焙：ExecuteBake → ConfigureCanvasScaler → 反序列化JSON → CreateUINode → ApplyComponentByType → 注册撤销操作。
  - 组件映射：根据节点类型选择UGUI组件并设置样式与行为。
  - 分辨率适配：根据配置选择目标分辨率，设置CanvasScaler的referenceResolution与matchWidthOrHeight。

```mermaid
flowchart TD
Start(["开始"]) --> Mode{"输入模式"}
Mode --> |文件| LoadFile["加载JSON文件"]
Mode --> |字符串| LoadString["读取字符串"]
LoadFile --> Parse["反序列化为UIDataNode"]
LoadString --> Parse
Parse --> Valid{"解析成功?"}
Valid --> |否| Error["输出错误并终止"]
Valid --> |是| CanvasCfg["配置CanvasScaler"]
CanvasCfg --> Bake["CreateUINode(root)"]
Bake --> Comp["ApplyComponentByType(go,nodeData)"]
Comp --> Children{"是否有子节点?"}
Children --> |是| Recurse["递归处理子节点"]
Children --> |否| Done["完成烘焙"]
Recurse --> Comp
```

**图表来源**
- [HtmlToUGUIBaker.cs:315-370](file://Assets/HtmlToUGUI/Editor/HtmlToUGUIBaker.cs#L315-L370)
- [HtmlToUGUIBaker.cs:394-421](file://Assets/HtmlToUGUI/Editor/HtmlToUGUIBaker.cs#L394-L421)
- [HtmlToUGUIBaker.cs:423-778](file://Assets/HtmlToUGUI/Editor/HtmlToUGUIBaker.cs#L423-L778)

**章节来源**
- [HtmlToUGUIBaker.cs:1-836](file://Assets/HtmlToUGUI/Editor/HtmlToUGUIBaker.cs#L1-L836)
- [HtmlToUGUIConfig.cs:1-35](file://Assets/HtmlToUGUI/HtmlToUGUIConfig.cs#L1-L35)
- [HtmlToUGUIConfig.asset:1-25](file://Assets/HtmlToUGUI/HtmlToUGUIConfig.asset#L1-L25)
- [test.json:1-1018](file://Assets/HtmlToUGUI/UIjson/test.json#L1-L1018)
- [HTML界面案例.html:1-146](file://Assets/HtmlToUGUI/HTML/HTML界面案例.html#L1-L146)

### UI事件处理与设置
- 事件处理：通过GameEventMgr提供的AddEvent系列方法订阅UI事件，支持不同参数类型的委托。
- 设置键名：Constant中提供UI相关设置键名（如UI音效开关与音量），便于统一管理。

**章节来源**
- [GameEventMgr.cs:48-94](file://Assets/TEngine/Runtime/Core/GameEvent/GameEventMgr.cs#L48-L94)
- [Constant.cs:1-21](file://Assets/TEngine/Runtime/Core/Constant/Constant.cs#L1-L21)

## 依赖关系分析
- UIBase与UIWindow：UIWindow继承自UIBase，获得通用能力并扩展窗口特有生命周期与渲染层管理。
- HtmlToUGUIBaker与HtmlToUGUIConfig：烘焙器依赖配置SO进行多分辨率与模板管理。
- UIWindow与UIModule：UIWindow通过UIModule进行资源加载、窗口栈管理与更新循环。
- GameEventMgr与UI：UI可通过事件系统订阅与派发事件，实现解耦交互。
- Constant与UI：UI设置键名集中管理，便于统一读取与持久化。

```mermaid
graph LR
UIBase --> UIWindow
HtmlToUGUIBaker --> HtmlToUGUIConfig
UIWindow --> UIModule
GameEventMgr --> UIWindow
Constant --> UIModule
```

**图表来源**
- [UIBase.cs:1-94](file://Assets/Launcher/Scripts/UIBase.cs#L1-L94)
- [UIWindow.cs:1-524](file://Assets/GameScripts/HotFix/GameLogic/Module/UIModule/UIWindow.cs#L1-L524)
- [HtmlToUGUIBaker.cs:1-836](file://Assets/HtmlToUGUI/Editor/HtmlToUGUIBaker.cs#L1-L836)
- [HtmlToUGUIConfig.cs:1-35](file://Assets/HtmlToUGUI/HtmlToUGUIConfig.cs#L1-L35)
- [GameEventMgr.cs:48-94](file://Assets/TEngine/Runtime/Core/GameEvent/GameEventMgr.cs#L48-L94)
- [Constant.cs:1-21](file://Assets/TEngine/Runtime/Core/Constant/Constant.cs#L1-L21)

**章节来源**
- [UIBase.cs:1-94](file://Assets/Launcher/Scripts/UIBase.cs#L1-L94)
- [UIWindow.cs:1-524](file://Assets/GameScripts/HotFix/GameLogic/Module/UIModule/UIWindow.cs#L1-L524)
- [HtmlToUGUIBaker.cs:1-836](file://Assets/HtmlToUGUI/Editor/HtmlToUGUIBaker.cs#L1-L836)
- [HtmlToUGUIConfig.cs:1-35](file://Assets/HtmlToUGUI/HtmlToUGUIConfig.cs#L1-L35)
- [GameEventMgr.cs:48-94](file://Assets/TEngine/Runtime/Core/GameEvent/GameEventMgr.cs#L48-L94)
- [Constant.cs:1-21](file://Assets/TEngine/Runtime/Core/Constant/Constant.cs#L1-L21)

## 性能考量
- 组件查找与缓存：建议在脚本生成器中缓存常用子节点与组件引用，避免频繁FindChild/GetComponent带来的开销。
- 更新循环：UIWindow的InternalUpdate会遍历子UIWidget并调用OnUpdate，尽量减少不必要的子节点数量或合并更新逻辑。
- Canvas与Raycaster：Visible切换会启用/禁用GraphicRaycaster，避免在不可见时仍进行射线检测。
- 分辨率与CanvasScaler：烘焙时统一配置CanvasScaler，确保不同分辨率下布局一致性，减少运行时计算。
- 资源加载：优先使用异步加载（InternalLoad的异步分支），避免主线程阻塞。

[本节为通用指导，不直接分析具体文件]

## 故障排查指南
- 窗口未显示或无交互：
  - 检查Visible与Interactable状态，确认Canvas图层是否正确设置。
  - 确认GraphicRaycaster已启用且未被遮挡。
- 窗口层级错乱：
  - 使用Depth属性调整sortingOrder，并确保子Canvas同步更新。
- 资源加载失败：
  - 确认资源路径与FromResources标记一致；异步加载时检查UIModule.Resource接口。
- HTML到UGUI烘焙失败：
  - 检查JSON格式是否符合UIDataNode规范；确认目标Canvas存在；检查分辨率配置与DSL模板。
- 事件未触发：
  - 确认通过GameEventMgr正确订阅事件；检查事件类型与委托签名匹配。

**章节来源**
- [UIWindow.cs:143-218](file://Assets/GameScripts/HotFix/GameLogic/Module/UIModule/UIWindow.cs#L143-L218)
- [UIWindow.cs:464-502](file://Assets/GameScripts/HotFix/GameLogic/Module/UIModule/UIWindow.cs#L464-L502)
- [HtmlToUGUIBaker.cs:315-370](file://Assets/HtmlToUGUI/Editor/HtmlToUGUIBaker.cs#L315-L370)
- [GameEventMgr.cs:48-94](file://Assets/TEngine/Runtime/Core/GameEvent/GameEventMgr.cs#L48-L94)

## 结论
TEngine UI系统通过清晰的层次划分与工具链支持，实现了从设计到运行时的高效闭环：UIBase与UIWindow提供了稳定的生命周期与渲染层管理；HtmlToUGUIBaker提升了UI搭建效率；GameEventMgr与Constant完善了事件与设置体系。遵循本文的最佳实践与扩展指南，可在保证性能的同时快速迭代UI功能。

[本节为总结性内容，不直接分析具体文件]

## 附录

### UI窗口管理机制（显示/隐藏、层级、遮罩）
- 显示/隐藏：通过Visible属性切换图层与Raycaster，实现显示与交互控制。
- 层级管理：通过Depth属性设置sortingOrder并同步子Canvas，保证层级一致性。
- 遮罩处理：通过Canvas与GraphicRaycaster共同控制交互范围，避免穿透点击。

**章节来源**
- [UIWindow.cs:143-218](file://Assets/GameScripts/HotFix/GameLogic/Module/UIModule/UIWindow.cs#L143-L218)
- [UIWindow.cs:494-497](file://Assets/GameScripts/HotFix/GameLogic/Module/UIModule/UIWindow.cs#L494-L497)

### UI生成器工具使用步骤
- 配置：创建或编辑HtmlToUGUIConfig，设置supportedResolutions与dslTemplateAsset。
- 输入：选择文件模式（拖拽JSON）或字符串模式（粘贴JSON）。
- 执行：为目标Canvas执行烘焙，工具将生成完整的UGUI树。
- 输出：在场景中得到可直接使用的UI层级结构。

**章节来源**
- [HtmlToUGUIConfig.cs:1-35](file://Assets/HtmlToUGUI/HtmlToUGUIConfig.cs#L1-L35)
- [HtmlToUGUIConfig.asset:1-25](file://Assets/HtmlToUGUI/HtmlToUGUIConfig.asset#L1-L25)
- [HtmlToUGUIBaker.cs:83-122](file://Assets/HtmlToUGUI/Editor/HtmlToUGUIBaker.cs#L83-L122)
- [HtmlToUGUIBaker.cs:315-370](file://Assets/HtmlToUGUI/Editor/HtmlToUGUIBaker.cs#L315-L370)

### UI开发最佳实践
- 组件绑定：在脚本生成器中缓存常用组件，减少运行时查找。
- 生命周期：合理使用OnCreate/OnRefresh/OnUpdate/OnDestroy，避免在不可见状态下执行昂贵操作。
- 事件处理：通过GameEventMgr统一订阅与派发，保持UI与业务解耦。
- 性能优化：优先异步加载资源；减少不必要的子节点；合理使用Canvas与Raycaster。
- 设置管理：使用Constant中的键名统一管理UI相关设置。

**章节来源**
- [UIWindow.cs:338-425](file://Assets/GameScripts/HotFix/GameLogic/Module/UIModule/UIWindow.cs#L338-L425)
- [GameEventMgr.cs:48-94](file://Assets/TEngine/Runtime/Core/GameEvent/GameEventMgr.cs#L48-L94)
- [Constant.cs:1-21](file://Assets/TEngine/Runtime/Core/Constant/Constant.cs#L1-L21)

### 扩展指南与自定义UI组件
- 自定义窗口：继承UIWindow，实现Init、ScriptGenerator、RegisterEvent、OnCreate/OnDestroy等。
- 自定义组件：在脚本生成器中添加新组件的绑定逻辑，并在ApplyComponentByType中映射到对应UGUI组件。
- 事件扩展：通过GameEventMgr新增事件类型与处理器，保持UI与模块解耦。

**章节来源**
- [UIWindow.cs:237-350](file://Assets/GameScripts/HotFix/GameLogic/Module/UIModule/UIWindow.cs#L237-L350)
- [HtmlToUGUIBaker.cs:423-778](file://Assets/HtmlToUGUI/Editor/HtmlToUGUIBaker.cs#L423-L778)
- [GameEventMgr.cs:48-94](file://Assets/TEngine/Runtime/Core/GameEvent/GameEventMgr.cs#L48-L94)