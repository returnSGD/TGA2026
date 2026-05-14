# 模块系统API

<cite>
**本文引用的文件**
- [Module.cs](file://Assets/TEngine/Runtime/Core/Module.cs)
- [ModuleSystem.cs](file://Assets/TEngine/Runtime/Core/ModuleSystem.cs)
- [RootModule.cs](file://Assets/TEngine/Runtime/Module/RootModule.cs)
- [ProcedureModule.cs](file://Assets/TEngine/Runtime/Module/ProcedureModule/ProcedureModule.cs)
- [IProcedureModule.cs](file://Assets/TEngine/Runtime/Module/ProcedureModule/IProcedureModule.cs)
- [ProcedureBase.cs](file://Assets/TEngine/Runtime/Module/ProcedureModule/ProcedureBase.cs)
- [FsmModule.cs](file://Assets/TEngine/Runtime/Module/FsmModule/FsmModule.cs)
- [IFsmModule.cs](file://Assets/TEngine/Runtime/Module/FsmModule/IFsmModule.cs)
- [TimerModule.cs](file://Assets/TEngine/Runtime/Module/TimerModule/TimerModule.cs)
- [ITimerModule.cs](file://Assets/TEngine/Runtime/Module/TimerModule/ITimerModule.cs)
</cite>

## 目录
1. [简介](#简介)
2. [项目结构](#项目结构)
3. [核心组件](#核心组件)
4. [架构总览](#架构总览)
5. [详细组件分析](#详细组件分析)
6. [依赖关系分析](#依赖关系分析)
7. [性能考虑](#性能考虑)
8. [故障排查指南](#故障排查指南)
9. [结论](#结论)
10. [附录](#附录)

## 简介
本文件为 TEngine 模块系统API的权威参考，覆盖以下模块：
- RootModule 根模块：负责全局初始化、帧驱动、全局设置（帧率、时间缩放、后台运行、休眠策略）、日志/文本/JSON辅助器注入、低内存回收等。
- ProcedureModule 流程模块：基于有限状态机的流程管理器，提供流程生命周期管理、状态切换、流程查询与重启等能力。
- FsmModule 状态机模块：通用有限状态机管理器，支持多持有者、多命名状态机，提供创建、查询、销毁与更新调度。
- TimerModule 计时器模块：统一计时器调度，支持循环/一次性、逻辑时间/真实时间两种模式，提供暂停/恢复/重置/移除等。

## 项目结构
模块系统采用“接口+实现”的分层设计，核心抽象位于 Core 层，模块实现位于 Module 子目录，各模块通过 ModuleSystem 统一注册、排序与轮询。

```mermaid
graph TB
subgraph "核心层"
M["Module 抽象类"]
MS["ModuleSystem 管理器"]
IU["IUpdateModule 接口"]
end
subgraph "模块层"
RM["RootModule 根模块"]
PM["ProcedureModule 流程模块"]
FM["FsmModule 状态机模块"]
TM["TimerModule 计时器模块"]
end
subgraph "接口层"
IPM["IProcedureModule 接口"]
IFM["IFsmModule 接口"]
ITM["ITimerModule 接口"]
end
M --> RM
M --> PM
M --> FM
M --> TM
MS --> RM
MS --> PM
MS --> FM
MS --> TM
IU --> RM
IU --> FM
IU --> TM
PM --> IPM
FM --> IFM
TM --> ITM
```

图示来源
- [Module.cs:22-39](file://Assets/TEngine/Runtime/Core/Module.cs#L22-L39)
- [ModuleSystem.cs:9-208](file://Assets/TEngine/Runtime/Core/ModuleSystem.cs#L9-L208)
- [RootModule.cs:10-304](file://Assets/TEngine/Runtime/Module/RootModule.cs#L10-L304)
- [ProcedureModule.cs:8-209](file://Assets/TEngine/Runtime/Module/ProcedureModule/ProcedureModule.cs#L8-L209)
- [FsmModule.cs:9-396](file://Assets/TEngine/Runtime/Module/FsmModule/FsmModule.cs#L9-L396)
- [TimerModule.cs:8-478](file://Assets/TEngine/Runtime/Module/TimerModule/TimerModule.cs#L8-L478)

章节来源
- [Module.cs:1-40](file://Assets/TEngine/Runtime/Core/Module.cs#L1-L40)
- [ModuleSystem.cs:1-208](file://Assets/TEngine/Runtime/Core/ModuleSystem.cs#L1-L208)

## 核心组件
- 模块抽象与轮询接口
  - Module：定义优先级、OnInit、Shutdown 抽象方法。
  - IUpdateModule：定义 Update(elapsed, realElapsed) 轮询接口。
- 模块系统
  - ModuleSystem：集中注册、排序、构建执行列表、统一 Update、统一 Shutdown。
  - 支持按接口类型获取模块、按类型创建模块、手动注册自定义模块实例。

章节来源
- [Module.cs:8-39](file://Assets/TEngine/Runtime/Core/Module.cs#L8-L39)
- [ModuleSystem.cs:9-208](file://Assets/TEngine/Runtime/Core/ModuleSystem.cs#L9-L208)

## 架构总览
模块系统以 RootModule 为入口，贯穿初始化、帧驱动、模块注册与轮询、资源回收与关闭。

```mermaid
sequenceDiagram
participant Game as "游戏主循环"
participant Root as "RootModule"
participant MS as "ModuleSystem"
participant Mod as "各模块(含Fsm/Timer)"
participant GC as "垃圾回收"
Game->>Root : "Awake/Start"
Root->>Root : "初始化文本/日志/JSON辅助器"
Root->>MS : "注册模块(按优先级插入)"
Root->>MS : "每帧调用 Update(delta, unscaled)"
MS->>Mod : "遍历 IUpdateModule 列表执行 Update"
Root->>GC : "OnLowMemory 触发回收"
Game->>Root : "OnDestroy"
Root->>MS : "Shutdown(逆序关闭)"
```

图示来源
- [RootModule.cs:116-167](file://Assets/TEngine/Runtime/Module/RootModule.cs#L116-L167)
- [ModuleSystem.cs:29-60](file://Assets/TEngine/Runtime/Core/ModuleSystem.cs#L29-L60)
- [ModuleSystem.cs:143-194](file://Assets/TEngine/Runtime/Core/ModuleSystem.cs#L143-L194)

## 详细组件分析

### RootModule 根模块 API
- 全局设置
  - 帧率：可读写，内部同步到 Application.targetFrameRate。
  - 游戏速度：可读写，内部同步到 Time.timeScale；提供 IsGamePaused、IsNormalGameSpeed 辅助判断。
  - 后台运行：可读写，同步到 Application.runInBackground。
  - 禁止休眠：可读写，根据值设置 Screen.sleepTimeout。
  - 编辑器语言：可读写（编辑器内有效）。
  - 文本/日志/JSON辅助器：通过类型名反射创建并注入。
- 生命周期与帧驱动
  - Awake：初始化辅助器、设置帧率/时间缩放/后台运行/休眠策略、注册低内存回调、启动 GameTime。
  - Update/FixedUpdate/LateUpdate：推进 GameTime 并调用 ModuleSystem.Update。
  - OnDestroy：在非编辑器环境调用 ModuleSystem.Shutdown。
  - OnApplicationQuit：注销低内存回调并停止协程。
- 低内存处理
  - 触发时尝试释放对象池未使用对象与资源模块缓存。
- 帧驱动与模块轮询
  - 每帧向 ModuleSystem 传递逻辑时间与真实时间，由其统一调度 IUpdateModule。

最佳实践
- 在 Awake 阶段完成辅助器与全局设置，确保模块初始化前可用。
- 使用 GameSpeed 控制全局节奏，配合 IsGamePaused 判断暂停状态。
- 将耗时任务拆分为多帧执行，避免单帧卡顿。

章节来源
- [RootModule.cs:30-111](file://Assets/TEngine/Runtime/Module/RootModule.cs#L30-L111)
- [RootModule.cs:116-167](file://Assets/TEngine/Runtime/Module/RootModule.cs#L116-L167)
- [RootModule.cs:287-302](file://Assets/TEngine/Runtime/Module/RootModule.cs#L287-L302)

### ProcedureModule 流程模块 API
- 模块特性
  - 优先级较低（-2），确保在其他模块之后轮询。
  - 内部持有 IFsmModule 与 IFsm<IProcedureModule>。
- 关键接口
  - Initialize：绑定 IFsmModule 并创建流程状态机，传入若干流程。
  - StartProcedure：按泛型或类型启动指定流程。
  - HasProcedure/GetProcedure：查询与获取流程实例。
  - CurrentProcedure/CurrentProcedureTime：获取当前流程及其已持续时间。
  - RestartProcedure：销毁旧状态机并用新流程集重建，随后启动首个流程。
  - Shutdown：销毁流程状态机并释放引用。
- 与 FsmModule 的协作
  - 通过 IFsmModule.CreateFsm 创建流程状态机，流程本身继承 FsmState<IProcedureModule>。

最佳实践
- 在应用启动阶段调用 Initialize 并立即 StartProcedure 启动首个流程。
- 使用 CurrentProcedureTime 记录流程阶段耗时，便于统计与调试。
- RestartProcedure 用于热更新或流程重置场景，需保证新流程集合有效。

章节来源
- [ProcedureModule.cs:8-209](file://Assets/TEngine/Runtime/Module/ProcedureModule/ProcedureModule.cs#L8-L209)
- [IProcedureModule.cs:8-83](file://Assets/TEngine/Runtime/Module/ProcedureModule/IProcedureModule.cs#L8-L83)
- [ProcedureBase.cs:8-59](file://Assets/TEngine/Runtime/Module/ProcedureModule/ProcedureBase.cs#L8-L59)

### FsmModule 状态机模块 API
- 模块特性
  - 优先级较高（1），确保在模块轮询早期更新。
  - 实现 IUpdateModule，对所有状态机进行统一 Update。
- 查询与统计
  - Count：返回状态机数量。
  - HasFsm：按持有者类型或（持有者类型+名称）查询是否存在状态机。
  - GetFsm：按持有者类型或（持有者类型+名称）获取状态机实例。
  - GetAllFsms：获取全部状态机数组或追加到结果列表。
- 创建与销毁
  - CreateFsm：支持无名/有名、集合/列表两种形式创建状态机。
  - DestroyFsm：支持按持有者类型/名称/实例销毁，返回是否成功。
- 更新调度
  - Update：遍历临时列表对未销毁状态机执行 Update，避免并发修改。

最佳实践
- 为复杂业务创建独立命名的状态机，便于隔离与调试。
- 在模块 Shutdown 中显式销毁状态机，避免残留引用。
- 对频繁创建销毁的状态机，考虑池化或延迟销毁策略。

章节来源
- [FsmModule.cs:9-396](file://Assets/TEngine/Runtime/Module/FsmModule/FsmModule.cs#L9-L396)
- [IFsmModule.cs:9-175](file://Assets/TEngine/Runtime/Module/FsmModule/IFsmModule.cs#L9-L175)

### TimerModule 计时器模块 API
- 模块特性
  - 实现 IUpdateModule，分别维护“逻辑时间”和“真实时间”两类计时器列表。
  - 提供统一 Update(elapsed, realElapsed) 调度。
- 计时器管理
  - AddTimer：添加计时器，支持循环、非缩放时间、回调参数。
  - Stop/Resume/IsRunning：暂停、恢复、查询运行状态。
  - GetLeftTime/Restart/Reset/ResetTimer：查询剩余时间、重置到初始状态、重置参数（含切换是否受时间缩放影响）。
  - RemoveTimer/RemoveAllTimer：按ID或全部移除。
- 内部机制
  - 使用两个有序列表分别存储逻辑/真实时间计时器，按剩余时间升序插入。
  - 支持“坏帧补偿”：当循环计时器在同帧内多次到期，递归触发回调直至不再到期。
  - 提供 AddSystemTimer 用于系统级定时器（如外部监控）。

最佳实践
- 优先使用 AddTimer 的参数化回调，避免闭包捕获导致的GC压力。
- 对 UI/动画等对时间缩放敏感的逻辑使用 isUnscaled=false（默认）；对调试/统计等不受影响的逻辑使用 isUnscaled=true。
- 大量短周期循环计时器应谨慎使用，必要时合并或降频。

章节来源
- [TimerModule.cs:8-478](file://Assets/TEngine/Runtime/Module/TimerModule/TimerModule.cs#L8-L478)
- [ITimerModule.cs:3-66](file://Assets/TEngine/Runtime/Module/TimerModule/ITimerModule.cs#L3-L66)

## 依赖关系分析
模块间依赖关系如下：

```mermaid
classDiagram
class Module {
+int Priority
+OnInit()
+Shutdown()
}
class IUpdateModule {
+Update(elapsed, realElapsed)
}
class ModuleSystem {
+GetModule~T~()
+RegisterModule~T~(module)
+Update(elapsed, realElapsed)
+Shutdown()
}
class RootModule {
+FrameRate : int
+GameSpeed : float
+RunInBackground : bool
+NeverSleep : bool
+PauseGame()
+ResumeGame()
+ResetNormalGameSpeed()
}
class ProcedureModule {
+Initialize(fsmModule, procedures)
+StartProcedure(type)
+HasProcedure(type)
+GetProcedure(type)
+CurrentProcedure
+CurrentProcedureTime : float
+RestartProcedure(procedures)
+Shutdown()
}
class FsmModule {
+CreateFsm(...)
+GetFsm(...)
+HasFsm(...)
+DestroyFsm(...)
+Update(elapsed, realElapsed)
+Shutdown()
}
class TimerModule {
+AddTimer(cb, time, loop, unscaled, args)
+Stop(id)
+Resume(id)
+IsRunning(id)
+GetLeftTime(id)
+Restart(id)
+ResetTimer(id,...)
+RemoveTimer(id)
+RemoveAllTimer()
+Update(elapsed, realElapsed)
+Shutdown()
}
Module <|-- RootModule
Module <|-- ProcedureModule
Module <|-- FsmModule
Module <|-- TimerModule
IUpdateModule <|.. RootModule
IUpdateModule <|.. FsmModule
IUpdateModule <|.. TimerModule
ModuleSystem --> RootModule : "注册/轮询"
ModuleSystem --> ProcedureModule : "注册/轮询"
ModuleSystem --> FsmModule : "注册/轮询"
ModuleSystem --> TimerModule : "注册/轮询"
ProcedureModule --> FsmModule : "依赖"
```

图示来源
- [Module.cs:22-39](file://Assets/TEngine/Runtime/Core/Module.cs#L22-L39)
- [ModuleSystem.cs:9-208](file://Assets/TEngine/Runtime/Core/ModuleSystem.cs#L9-L208)
- [RootModule.cs:10-304](file://Assets/TEngine/Runtime/Module/RootModule.cs#L10-L304)
- [ProcedureModule.cs:8-209](file://Assets/TEngine/Runtime/Module/ProcedureModule/ProcedureModule.cs#L8-L209)
- [FsmModule.cs:9-396](file://Assets/TEngine/Runtime/Module/FsmModule/FsmModule.cs#L9-L396)
- [TimerModule.cs:8-478](file://Assets/TEngine/Runtime/Module/TimerModule/TimerModule.cs#L8-L478)

## 性能考虑
- 模块轮询
  - ModuleSystem 通过链表维护模块顺序，按优先级插入；IUpdateModule 列表按需重建，避免频繁分配。
  - Update 时先检测脏标记，仅在变更时重建执行列表。
- 状态机
  - FsmModule 使用临时列表遍历，避免遍历时修改字典。
  - 按剩余时间有序插入，查找与更新效率高。
- 计时器
  - 逻辑/真实时间双列表分离，避免相互干扰。
  - 循环回调“坏帧补偿”递归处理，确保到期一致性。
- 内存与GC
  - 建议使用参数化回调，避免闭包捕获。
  - RootModule.OnLowMemory 主动触发对象池与资源模块回收，降低内存峰值风险。

## 故障排查指南
- 获取模块失败
  - 现象：按接口获取模块时报“必须按接口获取”或“找不到模块类型”。
  - 排查：确认模块实现类命名与接口命名一致，且程序集名称匹配；确保模块实现类可被反射创建。
- 未初始化错误
  - 现象：访问 CurrentProcedure 或调用 StartProcedure 前抛出“必须先初始化流程”。
  - 排查：先调用 Initialize 绑定 IFsmModule 并传入流程集合，再 StartProcedure。
- 状态机重复创建
  - 现象：创建同名状态机抛出“已存在”异常。
  - 排查：确保命名唯一，或先 DestroyFsm 再创建。
- 计时器无效
  - 现象：回调不触发或状态异常。
  - 排查：确认 IsRunning 状态、是否被标记移除、是否使用了正确的时间模式（逻辑/真实）。

章节来源
- [ModuleSystem.cs:68-89](file://Assets/TEngine/Runtime/Core/ModuleSystem.cs#L68-L89)
- [ProcedureModule.cs:86-95](file://Assets/TEngine/Runtime/Module/ProcedureModule/ProcedureModule.cs#L86-L95)
- [FsmModule.cs:240-250](file://Assets/TEngine/Runtime/Module/FsmModule/FsmModule.cs#L240-L250)
- [TimerModule.cs:340-386](file://Assets/TEngine/Runtime/Module/TimerModule/TimerModule.cs#L340-L386)

## 结论
TEngine 模块系统以清晰的抽象与严格的生命周期管理为核心，结合 RootModule 的全局驱动与 ModuleSystem 的统一调度，形成稳定高效的模块生态。流程模块、状态机模块与计时器模块分别覆盖“流程编排”“状态建模”“时间事件”三大关键领域，配合完善的接口与最佳实践，可满足从入门到复杂项目的开发需求。

## 附录
- 使用示例与最佳实践路径（以路径代替代码片段）
  - RootModule 全局设置与帧驱动
    - 设置帧率与时间缩放：[RootModule.cs:66-79](file://Assets/TEngine/Runtime/Module/RootModule.cs#L66-L79)
    - 后台运行与休眠策略：[RootModule.cs:94-111](file://Assets/TEngine/Runtime/Module/RootModule.cs#L94-L111)
    - 初始化辅助器与低内存处理：[RootModule.cs:214-285](file://Assets/TEngine/Runtime/Module/RootModule.cs#L214-L285)
  - ProcedureModule 流程管理
    - 初始化与启动流程：[ProcedureModule.cs:86-109](file://Assets/TEngine/Runtime/Module/ProcedureModule/ProcedureModule.cs#L86-L109)
    - 查询与重启流程：[ProcedureModule.cs:130-153](file://Assets/TEngine/Runtime/Module/ProcedureModule/ProcedureModule.cs#L130-L153), [ProcedureModule.cs:192-207](file://Assets/TEngine/Runtime/Module/ProcedureModule/ProcedureModule.cs#L192-L207)
  - FsmModule 状态机管理
    - 创建与销毁状态机：[FsmModule.cs:226-283](file://Assets/TEngine/Runtime/Module/FsmModule/FsmModule.cs#L226-L283), [FsmModule.cs:289-366](file://Assets/TEngine/Runtime/Module/FsmModule/FsmModule.cs#L289-L366)
    - 查询与遍历：[FsmModule.cs:138-183](file://Assets/TEngine/Runtime/Module/FsmModule/FsmModule.cs#L138-L183), [FsmModule.cs:189-217](file://Assets/TEngine/Runtime/Module/FsmModule/FsmModule.cs#L189-L217)
  - TimerModule 计时器管理
    - 添加与控制计时器：[TimerModule.cs:39-56](file://Assets/TEngine/Runtime/Module/TimerModule/TimerModule.cs#L39-L56), [TimerModule.cs:101-126](file://Assets/TEngine/Runtime/Module/TimerModule/TimerModule.cs#L101-L126)
    - 重置与移除：[TimerModule.cs:151-204](file://Assets/TEngine/Runtime/Module/TimerModule/TimerModule.cs#L151-L204), [TimerModule.cs:235-263](file://Assets/TEngine/Runtime/Module/TimerModule/TimerModule.cs#L235-L263)