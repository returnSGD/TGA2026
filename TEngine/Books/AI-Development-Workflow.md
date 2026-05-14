# AI 开发工作流指南

> 本文档介绍 TEngine 项目完整的 AI 辅助开发工作流，包含 tengine-dev skill 按需查询架构、任务等级分级机制、会话缓存策略，以及与 openspec / unity-skills 的集成使用方式。

**更新时间**: 2026-04-21

---

## 前提条件

### 必需工具

在开始使用 TEngine AI 开发工作流之前，请确保已安装以下工具：

#### 1. Claude (claude.ai/code)

**推荐使用 Claude Code** 作为主要的 AI 编程助手：

- 官方地址：[https://claude.ai/code](https://claude.ai/code)
- 支持本地文件操作、代码编辑、Unity Editor 自动化
- 集成 openspec、tengine-dev 等技能
- 提供上下文感知的 TEngine 开发指导

**安装方法**:
```bash
# 通过官网下载并安装
# Windows: https://claude.ai/code/download
# 安装后登录 Anthropic 账号即可使用
```

#### 2. ccswitch（API 密钥管理和反代）

**ccswitch** 是一个强大的 Claude API 管理工具，支持：
- 多个 API 密钥管理
- 反向代理配置
- 自动切换和负载均衡

#### 3. claude-mem（长期记忆库）★ 强烈推荐

**claude-mem** 是 Claude 的向量数据库插件，提供跨会话的长期记忆能力：

- 🧠 **向量数据库**: 自动存储项目知识和历史经验
- 🔍 **智能搜索**: 快速查找过去的解决方案
- 📚 **知识积累**: 越用越智能，持续学习项目特性
- 🔗 **无缝集成**: 自动在 Claude Code 中启用

**安装方法**:
```bash
/plugin marketplace add thedotmack/claude-mem
/plugin install claude-mem
```

**安装完后自动会启用**


### MCP工具

#### Unity-MCP（Unity Editor 自动化）

- 通过 REST API 自动化 Unity Editor 操作
- 已集成到 `unity-skills` 技能中
- 安装方法见 [unity-mcp-guide.md](skills/tengine-dev/references/unity-mcp-guide.md)

**unity-skills 触发词**：Unity、Unity Skills、in Unity、automate Unity、editor automation、创建脚本、场景分析、build scene、管理资源、Unity Editor 自动化、Unity 编辑器、Unity 技能、操作 Unity、全自动模式、半自动模式

---

## 目录

- [前提条件](#前提条件)
  - [必需工具](#必需工具)
  - [可选工具](#可选工具)
- [概述](#概述)
- [tengine-dev Skill 工作流](#tengine-dev-skill-工作流)
  - [整体流程总览](#整体流程总览)
  - [时序图一：规范获取流程](#时序图一规范获取流程)
  - [时序图二：会话内缓存机制](#时序图二会话内缓存机制)
  - [时序图三：并行多主题查询](#时序图三并行多主题查询)
  - [时序图四：规范冲突处理](#时序图四规范冲突处理)
  - [任务等级分级说明](#任务等级分级说明)
- [快速开始](#快速开始)
- [openspec 工作流](#openspec-工作流)
- [tengine-dev Skills](#tengine-dev-skills)
- [集成工作流](#集成工作流)
  - [完整开发流程图](#完整开发流程图)
  - [详细流程说明](#详细流程说明)
  - [不同场景的工作流](#不同场景的工作流)
  - [与 Claude Code 配合的最佳实践](#与-claude-code-配合的最佳实践)
- [最佳实践](#最佳实践)
  - [需求定义](#1-需求定义)
  - [代码规范](#2-代码规范)
  - [Unity Editor 自动化](#3-unity-editor-自动化)
  - [文档维护](#4-文档维护)
  - [团队协作](#5-团队协作)
  - [调试技巧](#6-调试技巧)
  - [性能优化](#7-性能优化建议)
  - [安全注意事项](#8-安全注意事项)

---

## 概述

TEngine 项目提供了一套完整的 AI 辅助开发工作流，由以下核心组件构成：

- **tengine-dev skill**: Claude Code 专用 TEngine 开发技能，从 `references/` 按需提供精炼规范
- **任务等级分级（L1-L4）**: 按任务复杂度决定查询深度，简单任务零开销
- **会话内缓存**: 同一主题在同一会话中只查询一次，后续任务复用
- **冲突标注**: 主动检测 references 与代码冲突，标注后以代码实现为准
- **openspec**: 规范驱动的变更管理工具
- **Unity-MCP**: Unity Editor 自动化操作工具

---

## tengine-dev Skill 工作流

### 整体流程总览

```mermaid
flowchart TD
    A([用户发起任务]) --> B{判断任务等级}

    B -->|L1 简单\ntypo/注释/日志| C[直接编写代码]
    B -->|L2 调用\n单一 API 修改| D[触发 tengine-dev skill\n只查该主题]
    B -->|L3 功能\n新功能/跨文件| E[触发 tengine-dev skill\n全量相关主题]
    B -->|L4 架构\n系统设计/重构| F[触发 tengine-dev skill\n并行多主题]

    D --> G{会话缓存命中?}
    E --> G
    F --> G

    G -->|命中| H[复用已有规范摘要]
    G -->|未命中| I[skill 读取 references/\n提炼规范指引]

    I --> L[输出代码/方案]
    H --> L
    C --> L

    L --> M{规范与代码冲突?}
    M -->|有冲突| N[标注冲突点\n记录到 .claude/memory/]
    M -->|无冲突| O([任务完成])
    N --> O
```

---

### 时序图一：规范获取流程

> **核心优势**：tengine-dev skill 直接从精炼的 `references/` 文档提取规范，无多余上下文噪声。

```mermaid
sequenceDiagram
    participant U as 用户
    participant M as 主 Agent (Claude)
    participant S as tengine-dev (skill)
    participant R as references/

    U->>M: 请实现背包 UI
    Note over M: 判断等级: L3 功能
    M->>S: 触发 skill<br/>查询: UIWindow规范 + 资源管理规范

    activate S
    S->>R: 读取 ui-development.md
    S->>R: 读取 resource-management.md
    S->>R: 读取 event-system.md
    Note over S: 提炼关键规范指引
    S-->>M: 返回规范摘要
    deactivate S

    M-->>U: 输出符合规范的代码
```

---

### 时序图二：会话内缓存机制

> **核心优势**：同一会话中相同主题只查询一次，后续任务直接复用，避免重复消耗。

```mermaid
sequenceDiagram
    participant U as 用户
    participant M as 主 Agent
    participant S as tengine-dev skill
    participant C as 会话缓存

    U->>M: 任务①: 实现登录界面 UI
    M->>S: 查询 UIWindow 规范
    S-->>M: 返回 UIWindow 规范摘要
    M->>C: 缓存: UIWindow 规范 ✅
    M-->>U: 输出登录界面代码

    U->>M: 任务②: 实现设置界面 UI
    M->>C: 检查缓存: UIWindow 规范
    C-->>M: 命中缓存 ✅ 直接复用
    Note over M: 无需重复触发 skill<br/>零等待，零额外消耗
    M-->>U: 输出设置界面代码

    U->>M: 任务③: 设置界面添加音效按钮
    M->>C: 检查缓存: UIWindow ✅ / Audio ❌
    C-->>M: UIWindow 命中，Audio 未命中
    M->>S: 仅补充查询 AudioModule 规范
    S-->>M: 返回 Audio 规范摘要
    M->>C: 缓存: Audio 规范 ✅
    M-->>U: 输出音效按钮代码
```

---

### 时序图三：并行多主题查询（L4 架构任务）

> **核心优势**：架构级任务并行查询多个主题，汇总后统一决策，大幅减少串行等待。

```mermaid
sequenceDiagram
    participant U as 用户
    participant M as 主 Agent
    participant S1 as tengine-dev #1
    participant S2 as tengine-dev #2
    participant S3 as tengine-dev #3

    U->>M: 设计战斗系统架构<br/>涉及: UI + 事件 + FSM + 资源

    Note over M: 判断等级: L4 架构<br/>并行触发多主题查询

    par 并行查询
        M->>S1: 查询 UIWindow + UIWidget 规范
        M->>S2: 查询 GameEvent 事件系统规范
        M->>S3: 查询 FSM 状态机 + 资源加载规范
    end

    S1-->>M: UI 规范摘要
    S2-->>M: 事件系统摘要
    S3-->>M: FSM + 资源摘要

    Note over M: 汇总三份摘要<br/>统一架构决策
    M-->>U: 输出完整战斗系统架构方案
```

---

### 时序图四：规范冲突处理

> **核心优势**：AI 主动检测 references 与代码的不一致，标注冲突并记录，以代码实现为最终依据。

```mermaid
sequenceDiagram
    participant M as 主 Agent
    participant S as tengine-dev skill
    participant Code as 项目代码
    participant Mem as .claude/memory/

    M->>S: 查询某 API 规范
    S-->>M: references 描述: API_X(param1, param2)

    M->>Code: 读取实际代码实现
    Code-->>M: 实际签名: API_X(param1, param2, param3)

    Note over M: 检测到冲突!<br/>references 描述与代码不符

    M->>Mem: 记录 problem_YYYY-MM-DD.md<br/>冲突详情 + 分析

    Note over M: 以代码实现为准<br/>在输出中标注差异

    M-->>U: 输出代码，并标注冲突点
```

---

### 任务等级分级说明

| 等级 | 判断标准 | 知识查询策略 |
|------|---------|-------------|
| **L1 简单** | typo 修正、注释修改、日志输出、单行变量改名（前提：不涉及框架 API 名称、UI 节点前缀、事件定义或资源路径） | ❌ 跳过查询，直接编码 |
| **L2 调用** | 调用已知 API、单一模块的局部修改 | ✅ 触发 `tengine-dev` skill（只查该主题） |
| **L3 功能** | 新功能开发、跨文件修改、新增 UI/资源/事件逻辑 | ✅ 触发 `tengine-dev` skill（全量相关主题） |
| **L4 架构** | 模块设计、系统重构、多模块协作、架构决策 | ✅ 触发 `tengine-dev` skill（并行多主题） |

> **判断原则**：宁可高估等级，不可低估——不确定时上调一级。

### 工作流快速参考

```
┌─────────────────────────────────────────────────────────┐
│                   TEngine AI 工作流                      │
├─────────────────────────────────────────────────────────┤
│  Step 0  判断任务等级 L1/L2/L3/L4                        │
│  Step 1  L1 直接编码                                     │
│         L2-L4 触发 tengine-dev skill 获取规范            │
│         （会话内缓存命中则直接复用，无需重复触发）        │
│  Step 2  基于规范输出代码/方案                            │
│  Step 3  若规范与代码冲突，标注冲突，记录到 .claude/memory/│
└─────────────────────────────────────────────────────────┘
```

详细规范请参考：[CLAUDE.md](../UnityProject/CLAUDE.md)

---

## 快速开始

### 5 分钟上手 TEngine AI 开发

```mermaid
graph LR
    A[安装 openspec] --> B[创建第一个提案]
    B --> C[开始开发]
    C --> D[完成归档]

    style A fill:#e1f5e1
    style B fill:#e1f0ff
    style C fill:#ffe1f0
    style D fill:#e1ffe1
```

#### 第一步：安装工具

```bash
# 安装 openspec
npm install -g @fission-ai/openspec@latest

# 在 TEngine 项目根目录初始化
cd I:\WorkSpace\TEngine
openspec init

# 选择 Claude Code 作为 AI 工具
```

#### 第二步：创建第一个需求

```bash
# 创建一个简单的需求提案
/opsx:propose "add-simple-button"
```

Claude 会自动生成：
- ✅ `proposal.md` - 需求描述
- ✅ `design.md` - 技术设计
- ✅ `specs/` - 详细规范
- ✅ `tasks.md` - 任务清单

#### 第三步：开始开发

```bash
# 审查生成的文档后，开始实施
/opsx:apply
```

Claude 会：
1. 读取所有 artifact 内容
2. 逐个完成 tasks.md 中的任务
3. 自动触发 tengine-dev 技能（如果涉及 TEngine 代码）
4. 自动使用 unity-skills（如果需要操作 Unity Editor）

#### 第四步：测试与归档

```bash
# 所有任务完成后归档
/opsx:archive

# 提交代码
git add .
git commit -m "feat: add simple button"
git push
```

### 典型工作流示例

```mermaid
sequenceDiagram
    participant Dev as 开发者
    participant Claude as Claude Code
    participant OpenSpec as OpenSpec
    participant Unity as Unity Editor

    Dev->>Claude: "帮我添加一个背包系统"
    Claude->>OpenSpec: /opsx:propose "add-inventory"
    OpenSpec-->>Claude: 生成 artifacts
    Claude->>Dev: 展示提案和设计

    Dev->>Claude: "看起来不错，开始实施"
    Claude->>OpenSpec: /opsx:apply
    OpenSpec-->>Claude: 加载所有上下文

    Claude->>Claude: 读取 tasks.md
    Claude->>Claude: 触发 tengine-dev 技能
    Claude->>Claude: 生成 UIInventory.cs

    Claude->>Unity: 使用 unity-skills
    Unity-->>Claude: 创建 UI Prefab

    Claude->>Dev: "已完成所有任务，请测试"

    Dev->>Unity: 测试功能
    Unity-->>Dev: ✅ 功能正常

    Dev->>Claude: "测试通过，归档"
    Claude->>OpenSpec: /opsx:archive
    OpenSpec-->>Dev: ✅ 变更已归档
```

### 常见场景速查

| 场景 | 命令 | 说明 |
|------|------|------|
| 需求不明确 | `/opsx:explore` | 进入探索模式，梳理需求 |
| 新功能开发 | `/opsx:propose "feature-name"` | 创建提案并生成文档 |
| 开始编码 | `/opsx:apply` | 开始实施任务 |
| 查找历史经验 | `"上次怎么做的？"` | 触发 claude-mem 搜索 |
| Unity 自动化 | `"帮我创建..."`| 自动触发 unity-skills |
| 完成归档 | `/opsx:archive` | 归档变更 |

---

## openspec 工作流

### 什么是 openspec？

OpenSpec 是一个为AI编码助手设计的规范驱动开发工具，它通过轻量级的工作流程，确保人类开发者和AI助手在编写任何代码之前就能对需求达成明确共识。它通过以下方式帮助团队：

- 创建标准化的提案 (Proposal)
- 生成技术设计文档 (Design)
- 定义详细的规范 (Specs)
- 跟踪实施任务 (Tasks)

核心特点
- 🚀 轻量级：无需API密钥，最小化设置
- 🔄 现有项目优先：特别适合修改现有功能 (1→n)
- 📋 变更跟踪：提案、任务和规范差异的完整生命周期管理
- 🤖 AI工具集成：支持多种主流AI编码助手
2.为什么需要OpenSpec？
传统AI编码助手的问题
当您使用AI编码助手时，是否遇到过以下情况：
- ❌ AI根据模糊提示生成不符合需求的代码
- ❌ 遗漏重要功能要求
- ❌ 添加了不必要的功能
- ❌ 代码行为不可预测
OpenSpec的解决方案
OpenSpec通过规范驱动开发解决这些问题：
✅ 明确共识：在编码前确定所有要求
✅ 结构化变更：所有相关文档集中管理
✅ 可审查输出：AI根据明确规范生成代码
✅ 版本控制：完整追踪所有变更历史

### 安装

```bash
npm install -g @fission-ai/openspec@latest

cd your-project-directory

openspec init
```
初始化过程会：
1. 询问您使用的AI工具（Claude Code、Cursor等）
2. 自动配置相应的斜杠命令
3. 创建 openspec/ 目录结构

### 基础命令

```bash
# 新建议题
/opsx:propose 新建议题我要做什么

# 开始执行
/opsx:apply 我还要附加什么条件

# 修改完成手动归档
/opsx:archive

# 获取指令帮助
openspec instructions <artifact-id> --change "my-feature"
```

### 推荐配合claude-mem 内有向量数据库长久记忆！

### 变更结构

每个变更包含以下 artifacts：

| Artifact | 描述 |
|----------|------|
| proposal.md | 变更提案，说明"为什么"和"做什么" |
| design.md | 技术设计文档，说明"如何实现" |
| specs/*.md | 详细规范，定义系统行为 |
| tasks.md | 实施任务清单 |

---

## opsx:explore 探索模式

### 什么是探索模式？

`/opsx:explore` 是 openspec 工作流中的**思考伙伴模式**，用于在需求不明确、问题复杂、或需要调查现有代码时，先进行深度探索再决定行动方向。它不是直接生成代码，而是帮你理清思路、发现问题、评估方案。

### 核心定位

```
┌──────────────────────────────────────────────────────────────┐
│              openspec 工作流四阶段                             │
├──────────────────────────────────────────────────────────────┤
│  1. /opsx:explore  →  探索思考，梳理需求和约束               │
│  2. /opsx:propose  →  创建提案，生成设计文档和任务清单         │
│  3. /opsx:apply    →  实施开发，逐个完成任务                  │
│  4. /opsx:archive  →  归档变更，更新主规范                    │
└──────────────────────────────────────────────────────────────┘
```

> **关键区别**：`explore` 是**只读探索**，不修改任何文件；`propose` 是**写入提案**，创建变更目录和 artifacts。

### 何时使用 explore？

| 场景 | 说明 | 示例 |
|------|------|------|
| **需求模糊** | 只有一个想法，不确定具体要做什么 | "我想优化 UI 加载性能" |
| **问题调查** | 出现 Bug 或异常，需要先定位根因 | "战斗 UI 偶尔崩溃" |
| **方案对比** | 有多种实现路径，需要评估利弊 | "用 FSM 还是状态模式管理角色状态？" |
| **代码理解** | 需要深入理解某模块再决定如何修改 | "资源模块的引用计数是怎么工作的？" |
| **重构评估** | 想重构但不确定影响范围 | "把事件系统从 int 改为接口事件" |
| **架构决策** | 涉及多模块协作，需要全局视角 | "设计跨模块的战斗系统架构" |

### explore 的工作方式

```mermaid
flowchart TD
    Start([/opsx:explore]) --> Input{输入类型}

    Input -->|问题/疑问| Investigate[调查问题]
    Input -->|想法/需求| Brainstorm[头脑风暴]
    Input -->|代码理解| Analyze[代码分析]

    Investigate --> SearchCode[搜索相关代码]
    SearchCode --> ReadImpl[读取实现细节]
    ReadImpl --> IdentifyRoot[识别根因/关键点]

    Brainstorm --> ListOptions[列出可行方案]
    ListOptions --> CompareTradeoffs[对比利弊]

    Analyze --> TraceFlow[追踪调用链]
    TraceFlow --> MapDeps[映射依赖关系]

    IdentifyRoot --> Synthesize[综合分析]
    CompareTradeoffs --> Synthesize
    MapDeps --> Synthesize

    Synthesize --> Output[输出探索结论]
    Output --> Decision{下一步?}

    Decision -->|需求已明确| Propose[/opsx:propose]
    Decision -->|还需深入| Start
    Decision -->|直接修复| DirectFix[直接编写修复代码]

    style Start fill:#e1f0ff
    style Propose fill:#e1ffe1
    style DirectFix fill:#fff4e1
```

### explore 与 tengine-dev 的协作

在探索过程中，Claude 会自动触发 `tengine-dev` skill 获取框架规范，确保探索结论符合 TEngine 架构约束：

```mermaid
sequenceDiagram
    participant U as 用户
    participant M as 主 Agent
    participant E as /opsx:explore
    participant S as tengine-dev skill
    participant Code as 项目代码

    U->>M: "我想优化 UI 加载性能，但不确定瓶颈在哪"
    M->>E: 进入探索模式

    E->>S: 查询资源加载规范 + UI 生命周期规范
    S-->>E: 返回 TEngine 规范摘要

    E->>Code: 搜索 UIWindow 子类中的资源加载模式
    E->>Code: 检查 SetSprite vs LoadAssetAsync<Sprite> 使用情况
    E->>Code: 分析 OnCreate/OnRefresh 中的 await 调用

    Note over E: 综合分析：<br/>1. 3个UIWindow用了LoadAssetAsync<Sprite>（应改SetSprite）<br/>2. OnCreate中有串行await（应改并行）<br/>3. 缺少CancellationToken取消支持

    E-->>M: 输出探索结论 + 优化建议
    M-->>U: 展示分析结果和方案选项

    U->>M: "方案1不错，创建提案"
    M->>M: /opsx:propose "optimize-ui-loading"
```

### 使用示例

#### 示例 1：Bug 调查

```
用户：/opsx:explore
      "战斗界面偶尔出现 NullReferenceException，
       堆栈指向 BattleMainUI.OnRefresh"

Claude 探索过程：
1. 读取 BattleMainUI.cs → 发现 OnRefresh 访问 _playerData
2. 搜索 _playerData 赋值位置 → 仅在 OnCreate 中赋值
3. 分析生命周期 → OnCreate 只执行一次，OnRefresh 每次都执行
4. 查找事件流 → GameEvent.Send(OnPlayerDataChanged) 可能在 OnCreate 之前触发
5. 结论：_playerData 可能在首次 OnRefresh 时尚未初始化

输出：
- 根因：_playerData 在 OnCreate 中赋值，但 ShowUI 可能在资源加载完成前触发 OnRefresh
- 建议修复：在 OnRefresh 中添加 null 检查，或在 OnCreate 中初始化默认值
- 下一步：/opsx:propose "fix-battle-ui-null-ref"
```

#### 示例 2：架构评估

```
用户：/opsx:explore
      "我想用接口事件替换所有 int 事件常量，
       需要评估工作量和风险"

Claude 探索过程：
1. 搜索所有 GameEvent.Send(int, ...) 调用 → 找到 23 处
2. 搜索所有 AddUIEvent(int, ...) 调用 → 找到 15 处
3. 搜索现有 [EventInterface] 接口 → 找到 3 个已迁移
4. 读取 event-system.md 规范 → 确认接口事件模式
5. 分析依赖 → 识别出 5 个模块需要同步修改

输出：
- 现状：23 处 int 事件发送 + 15 处 UI 监听，3 个接口已迁移
- 方案A：渐进式迁移（按模块分批，低风险，2-3天）
- 方案B：一次性迁移（需全面测试，高风险，1天）
- 建议：方案A，先迁移事件定义最集中的 IBattleEvent
- 下一步：/opsx:propose "migrate-to-interface-events"
```

#### 示例 3：性能瓶颈定位

```
用户：/opsx:explore
      "背包界面打开要 2 秒，太慢了"

Claude 探索过程：
1. 读取 BagUI.cs → OnCreate 中串行 await 5 次 LoadAssetAsync
2. 检查资源大小 → 5 个 Sprite 均小于 100KB
3. 分析 SetSprite 使用 → 未使用，手动 LoadAssetAsync<Sprite>
4. 对比规范 → SetSprite 有缓存池，LoadAssetAsync<Sprite> 没有
5. 结论：串行加载 + 未使用缓存池是主要瓶颈

输出：
- 瓶颈1：5 个串行 await → 改 UniTask.WhenAll 并行加载
- 瓶颈2：LoadAssetAsync<Sprite> → 改 SetSprite（自带缓存池）
- 预期提升：2s → ~400ms（5x 并行 + 缓存命中）
- 下一步：/opsx:propose "optimize-bag-ui-loading"
```

### explore 最佳实践

1. **先探索再提案**：任何涉及 3 个以上文件或不确定影响的任务，先用 explore
2. **带着问题探索**：给 explore 明确的疑问，而非泛泛的"帮我看看"
3. **探索结论驱动决策**：explore 的输出应直接决定是否 propose、如何 propose
4. **探索可迭代**：如果首次探索不够深入，可以继续追问，无需退出
5. **探索不修改代码**：explore 是只读的，不会修改任何文件（这是与 apply 的核心区别）

### explore → propose → apply 完整流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant C as Claude Code

    Note over U,C: 阶段1：探索（只读）
    U->>C: /opsx:explore "需求描述"
    C->>C: 搜索代码 + 读取实现 + 查询规范
    C-->>U: 输出探索结论（问题/方案/影响分析）

    Note over U,C: 阶段2：提案（写入文档）
    U->>C: /opsx:propose "change-name"
    C->>C: 生成 proposal.md + design.md + specs/ + tasks.md
    C-->>U: 展示提案，等待审查

    Note over U,C: 阶段3：实施（写入代码）
    U->>C: /opsx:apply
    C->>C: 逐个完成 tasks.md 中的任务
    C-->>U: 所有任务完成

    Note over U,C: 阶段4：归档
    U->>C: /opsx:archive
    C-->>U: 变更已归档，规范已更新
```

---

## tengine-dev Skills

### 什么是 tengine-dev？

`tengine-dev` 是 Claude Code 的技能，专门用于 TEngine 框架开发指导。

### 触发条件

在 TEngine 项目中编写或修改代码时，以下关键词会触发 tengine-dev 技能：

- **模块系统**: ResourceModule, AudioModule, TimerModule, GameModule
- **UI 开发**: UIWindow, UIWidget, UIModule
- **事件系统**: GameEvent, EventInterface, AddUIEvent
- **资源管理**: YooAsset, LoadAssetAsync, UnloadAsset
- **热更代码**: HybridCLR, GameApp, HotFix
- **配置表**: Luban, ConfigSystem

### 核心原则

tengine-dev 技能遵循以下核心原则：

1. **异步优先**: IO 操作用 `UniTask`，禁止同步加载/Coroutine
2. **模块访问**: 通过 `GameModule.XXX` 访问
3. **资源必须释放**: `LoadAssetAsync` 对应 `UnloadAsset`
4. **热更边界**: `GameScripts/Main` 不热更，`GameScripts/HotFix/` 全部热更
5. **事件解耦**: 模块间用 `GameEvent`，UI 内部用 `AddUIEvent`

### 程序集分层

```
GameScripts/Main/       → 主包（不热更）
GameScripts/HotFix/
  ├── GameProto/        → Luban 配置代码
  └── GameLogic/        → 业务逻辑（GameApp.cs 入口）
```

### 其他 Skills

#### luban-dev

Luban 游戏配置全栈工具，支持枚举/Bean/数据表的增删改查、代码生成、TEngine 集成。

**触发场景**：
- 编辑游戏配置数据（配置表/数据表/道具表/技能表/奖励表/活动表）
- 新增/修改/删除配置表结构
- 定义枚举/Bean/字段
- 导表/生成配置代码
- 编写 luban.conf 或 Schema 定义
- Luban 类型系统/校验器问题

> 即使用户未明确说"Luban"，只要是编辑游戏配置数据，也应使用此技能。

#### html-to-ugui

HTML 原型转 Unity UGUI 界面生成管线。通过 AI 生成符合 UI-DSL 规范的 HTML，烘焙为 JSON 坐标数据，再导入 Unity 自动生成 UGUI 节点树。

**触发场景**：
- 需要快速生成 Unity UGUI 界面原型
- 用自然语言描述 UI 需求并自动生成
- 创建 UIWindow/面板的初始布局
- 批量生成表单、设置、列表等标准界面

#### wiki-synchelper

Wiki 同步助手，用于"项目实现内容"与"开发 Wiki 文档"之间的双向同步，确保 AI 可基于 Wiki 快速理解项目现状并按规范继续开发。

**触发场景**：
- 用户要求扫描/比对/同步/报告项目与 Wiki 的差异
- 代码实现已更新但 Wiki 文档未跟进
- Wiki 文档需要反向修正代码结构
- 用户说"同步 Wiki"、"更新文档"、"Wiki 和代码不一致"、"扫描文档差异"

---

## 集成工作流

### 完整开发流程图

```mermaid
flowchart TD
    Start([开始新需求]) --> Explore{需求是否<br/>明确?}

    Explore -->|不明确| ExploreMode[/opsx:explore<br/>探索模式思考]
    ExploreMode --> Propose
    Explore -->|明确| Propose[/opsx:propose<br/>创建提案]

    Propose --> GenArtifacts[自动生成 artifacts<br/>proposal/design/specs/tasks]
    GenArtifacts --> Review{人工审查<br/>artifacts}

    Review -->|需要修改| EditArtifacts[手动编辑<br/>proposal.md/design.md等]
    EditArtifacts --> Review

    Review -->|通过| Apply[/opsx:apply<br/>开始实施]

    Apply --> LoadContext[加载变更上下文<br/>所有 artifacts]
    LoadContext --> ChooseTask[选择下一个任务<br/>从 tasks.md]

    ChooseTask --> CheckType{任务类型?}

    CheckType -->|代码开发| TengineDev[tengine-dev 技能<br/>TEngine 开发指导]
    CheckType -->|Unity 操作| UnitySkills[unity-skills<br/>Editor 自动化]
    CheckType -->|通用开发| GeneralDev[Claude Code<br/>常规开发]

    TengineDev --> CodeImpl[实现代码<br/>遵循 TEngine 规范]
    UnitySkills --> UnityImpl[执行 Unity 操作<br/>场景/Prefab/组件]
    GeneralDev --> CodeImpl
    UnityImpl --> CodeImpl

    CodeImpl --> Test[测试验证]
    Test --> TestResult{测试通过?}

    TestResult -->|失败| Debug[调试修复]
    Debug --> CodeImpl

    TestResult -->|通过| UpdateTask[更新 tasks.md<br/>标记完成]
    UpdateTask --> MemSave{重要经验?}

    MemSave -->|是| SaveMem[保存到 claude-mem<br/>跨会话记忆]
    MemSave -->|否| CheckMore
    SaveMem --> CheckMore{还有任务?}

    CheckMore -->|有| ChooseTask
    CheckMore -->|无| FinalTest[完整测试]

    FinalTest --> FinalResult{全部通过?}
    FinalResult -->|失败| ChooseTask
    FinalResult -->|通过| Archive[/opsx:archive<br/>归档变更]

    Archive --> CommitPush[git commit & push]
    CommitPush --> End([完成])

    style Start fill:#e1f5e1
    style End fill:#e1f5e1
    style Explore fill:#fff4e1
    style ExploreMode fill:#e1f0ff
    style TengineDev fill:#ffe1f0
    style UnitySkills fill:#f0e1ff
    style Archive fill:#e1ffe1
```

### 详细流程说明

#### 阶段 1: 需求探索与提案创建

```bash
# 如果需求不明确，先进入探索模式
/opsx:explore
# 在探索模式中思考：
# - 这个需求的真正目标是什么？
# - 有哪些技术方案可选？
# - 需要考虑哪些边界情况？

# 需求明确后，创建提案
/opsx:propose "add-user-inventory-system"
```

**自动生成的 artifacts**:
- `proposal.md` - 需求描述和目标
- `design.md` - 技术设计方案
- `specs/*.md` - 详细规范文档
- `tasks.md` - 实施任务清单

#### 阶段 2: 人工审查与完善

在 `openspec/changes/<change-name>/` 目录中检查生成的文档：

```bash
# 检查生成的文档
ls openspec/changes/add-user-inventory-system/

# 如果需要修改，直接编辑文件
# 修改后无需额外命令，opsx:apply 会自动加载最新内容
```

#### 阶段 3: 实施开发

```bash
# 开始实施，可以附加额外条件
/opsx:apply "使用 UniTask 异步加载，遵循 TEngine 资源管理规范"
```

**Claude Code 会自动**:
1. 加载所有 artifact 内容作为上下文
2. 读取 `tasks.md` 中的任务列表
3. 逐个实施任务

**开发过程中**:

- **TEngine 代码开发** → 自动触发 `tengine-dev` 技能
  - 提供模块系统、UI 开发、事件系统、资源管理等指导
  - 确保代码符合 TEngine 规范（异步优先、资源释放、热更边界等）

- **Unity Editor 操作** → 使用 `unity-skills`
  - 创建/修改 GameObject、Prefab、Scene
  - 配置组件、材质、动画
  - UI 布局和组件绑定

- **通用代码开发** → Claude Code 常规能力
  - C# 代码编写
  - 算法实现
  - 工具脚本

#### 阶段 4: 测试与验证

每完成一个任务后：
1. 运行相关测试
2. 在 Unity 中验证功能
3. 更新 `tasks.md` 标记完成状态

#### 阶段 5: 记忆保存（可选）

如果遇到重要问题或发现有价值的经验：

```bash
# Claude Code 会自动使用 claude-mem 保存
# 无需手动操作，AI 会判断是否值得记录
```

**会保存的内容**:
- 解决的技术难题
- TEngine 特定的最佳实践
- 常见错误的解决方案
- 项目特定的约定

#### 阶段 6: 归档与提交

```bash
# 所有任务完成并测试通过后
/opsx:archive

# 提交代码
git add .
git commit -m "feat: add user inventory system"
git push
```

### 不同场景的工作流

#### 场景 1: 新功能开发

```mermaid
flowchart LR
    A[/opsx:propose] --> B[编写 specs]
    B --> C[/opsx:apply]
    C --> D[tengine-dev 指导]
    D --> E[unity-skills 操作]
    E --> F[测试]
    F --> G[/opsx:archive]
```

```bash
# 示例：添加背包系统
/opsx:propose "add-inventory-system"
# 审查生成的 artifacts
/opsx:apply "使用 TEngine 模块系统"
# 开发...
/opsx:archive
```

#### 场景 2: Bug 修复

```mermaid
flowchart LR
    A[/opsx:explore] --> B[定位问题]
    B --> C[/opsx:propose]
    C --> D[/opsx:apply]
    D --> E[修复验证]
    E --> F[/opsx:archive]
```

```bash
# 示例：修复资源泄漏
/opsx:explore  # 先调查问题根源
/opsx:propose "fix-resource-leak-in-ui"
/opsx:apply
/opsx:archive
```

#### 场景 3: 重构优化

```mermaid
flowchart LR
    A[/opsx:explore] --> B[分析现状]
    B --> C[/opsx:propose]
    C --> D[设计新方案]
    D --> E[/opsx:apply]
    E --> F[渐进式重构]
    F --> G[测试对比]
    G --> H[/opsx:archive]
```

```bash
# 示例：优化 UI 加载性能
/opsx:explore  # 分析性能瓶颈
/opsx:propose "optimize-ui-loading"
# 在 design.md 中详细设计优化方案
/opsx:apply
/opsx:archive
```

#### 场景 4: Unity Editor 自动化操作

```mermaid
flowchart LR
    A[需求] --> B[/opsx:propose]
    B --> C[/opsx:apply]
    C --> D[unity-skills]
    D --> E[批量操作]
    E --> F[验证结果]
    F --> G[/opsx:archive]
```

```bash
# 示例：批量创建 UI Prefab
/opsx:propose "create-ui-panels-batch"
/opsx:apply "使用 unity-skills 批量创建 10 个 UI 面板预制体"
# AI 会调用 unity-skills 自动化操作 Unity Editor
/opsx:archive
```

### 与 Claude Code 配合的最佳实践

#### 1. 利用技能自动触发

**无需手动调用技能**，只需在描述中包含关键词：

```
❌ 不好：请使用 tengine-dev 技能帮我创建一个 UIWindow
✅ 好的：帮我创建一个背包 UIWindow，需要加载道具图标资源
→ 自动触发 tengine-dev，提供 UIWindow、资源管理指导
```

#### 2. 分阶段处理复杂任务

```bash
# 第一阶段：UI 框架
/opsx:propose "inventory-ui-framework"
/opsx:apply
/opsx:archive

# 第二阶段：数据逻辑
/opsx:propose "inventory-data-logic"
/opsx:apply
/opsx:archive

# 第三阶段：网络同步
/opsx:propose "inventory-network-sync"
/opsx:apply
/opsx:archive
```

#### 3. 充分利用 claude-mem

```
"上次我们是怎么处理 UI 资源释放的？"
→ 触发 claude-mem:mem-search，查找历史经验

"帮我探索一下如何实现技能系统"
→ 使用 claude-mem:smart-explore 高效扫描代码结构
```

#### 4. Unity Editor 操作自动化

```
"帮我在场景中创建一个包含 10 个道具槽位的背包 UI"
→ 自动触发 unity-skills
→ 调用 manage_ui、manage_gameobject 等工具
→ 直接在 Unity Editor 中创建完整结构
```

### 常用命令速查

| 操作 | 命令 |
|------|------|
| 探索需求 | `/opsx:explore` |
| 创建变更 | `/opsx:propose "<name>"` |
| 开始实施 | `/opsx:apply "额外条件"` |
| 归档完成 | `/opsx:archive` |
| 查看帮助 | `openspec instructions <artifact-id>` |

---

## 最佳实践

### 1. 需求定义

#### 使用探索模式澄清模糊需求

```bash
# 场景：需求不明确时
/opsx:explore

# 在探索模式中，Claude 会帮你思考：
# - 需求的核心价值是什么？
# - 有哪些隐含的边界条件？
# - 是否有现成的解决方案？
# - 需要考虑哪些技术约束？
```

#### 编写清晰的 Specs

在 `specs/*.md` 中使用场景驱动描述：

```markdown
## 场景 1: 用户打开背包

**前置条件**:
- 用户已登录
- 拥有至少 1 个道具

**操作流程**:
1. 点击主界面的背包按钮
2. 异步加载背包 UI Prefab
3. 从服务器获取道具列表
4. 渲染道具图标（异步加载）

**预期结果**:
- 背包界面在 1 秒内打开
- 所有道具图标正确显示
- 支持滚动查看更多道具

**异常情况**:
- 网络超时：显示错误提示
- 资源加载失败：使用默认图标
```

### 2. 代码规范

#### TEngine 核心原则检查清单

在每次开发完成后，确认：

- [ ] **异步优先**: 所有 IO 操作使用 `UniTask`
  ```csharp
  ✅ await GameModule.Resource.LoadAssetAsync<Sprite>(path);
  ❌ Resources.Load<Sprite>(path);  // 禁止
  ❌ StartCoroutine(LoadSprite());   // 禁止
  ```

- [ ] **资源释放**: 每个 `LoadAssetAsync` 都有对应的 `UnloadAsset`
  ```csharp
  ✅ private AssetHandle _handle;
      _handle = await GameModule.Resource.LoadAssetAsync<Sprite>(path);
      // 使用完毕后
      GameModule.Resource.UnloadAsset(_handle);

  ❌ await GameModule.Resource.LoadAssetAsync<Sprite>(path);
      // 没有保存 handle，无法释放
  ```

- [ ] **模块访问**: 通过 `GameModule.XXX` 访问
  ```csharp
  ✅ GameModule.Audio.PlaySound("click");
  ❌ ModuleSystem.GetModule<AudioModule>().PlaySound("click");
  ```

- [ ] **热更边界**: 代码放在正确的目录
  ```
  ✅ GameScripts/HotFix/GameLogic/UI/UIInventory.cs
  ❌ GameScripts/Main/UI/UIInventory.cs  // 不热更，错误！
  ```

- [ ] **事件解耦**: 使用正确的事件类型
  ```csharp
  ✅ 模块间：GameEvent.Send(EventName.OnInventoryChanged);
  ✅ UI 内：AddUIEvent(btnClose, OnCloseClicked);
  ❌ 直接引用：otherModule.OnInventoryChanged();
  ```

### 3. Unity Editor 自动化

#### 使用 unity-skills 提高效率

**批量创建 UI 结构**:
```
"帮我创建一个背包 UI，包含：
- 顶部标题栏（包含标题文本和关闭按钮）
- 中间 ScrollView（网格布局，3列）
- 底部按钮区（整理按钮、出售按钮）"

→ AI 自动调用 unity-skills 创建完整结构
```

**批量配置组件**:
```
"帮我给场景中所有名称包含 'Item' 的 GameObject 添加 BoxCollider 组件"

→ AI 批量操作，节省大量时间
```

**资源导入设置**:
```
"帮我将 Assets/UI/Icons 目录下的所有图片设置为：
- Texture Type: Sprite (2D and UI)
- Max Size: 512
- Compression: High Quality"

→ AI 批量修改导入设置
```

### 4. 文档维护

#### 及时更新项目文档

```bash
# 发现新的最佳实践时
# 直接告诉 Claude：
"这个资源释放的方案很好用，请记住：
在 UIWindow 中使用 List<AssetHandle> 统一管理所有加载的资源，
在 OnDestroy 时批量释放"

→ Claude 会自动保存到 claude-mem
→ 未来遇到类似场景会自动应用
```

#### 记录常见问题

创建 `troubleshooting.md` 记录解决方案：

```markdown
### 问题：UI 加载后资源未释放

**现象**: 内存持续增长，Unity Profiler 显示大量未释放的 Sprite

**原因**:
1. 忘记保存 AssetHandle
2. 在 OnDestroy 之前就销毁了窗口

**解决方案**:
使用 List<AssetHandle> 统一管理
...
```

### 5. 团队协作

#### 统一使用 openspec

```bash
# 团队成员 A：创建需求
/opsx:propose "add-friend-system"
git add openspec/
git commit -m "docs: add friend system proposal"
git push

# 团队成员 B：审查需求
git pull
# 查看 openspec/changes/add-friend-system/
# 如有修改，直接编辑 proposal.md 等文件
git commit -m "docs: refine friend system design"
git push

# 团队成员 C：实施开发
git pull
/opsx:apply
# 开发...
/opsx:archive
git push
```

#### Code Review 检查点

提交 PR 前的自检清单：

- [ ] `tasks.md` 中的所有任务已完成
- [ ] 代码符合 TEngine 规范（异步、资源释放等）
- [ ] 已在 Unity 中测试功能
- [ ] 重要改动已记录到文档
- [ ] 变更已归档（`/opsx:archive`）

### 6. 调试技巧

#### 利用 Claude 快速定位问题

```
"为什么我的 UI 加载后背景图显示不出来？
代码在 GameScripts/HotFix/GameLogic/UI/UIInventory.cs"

→ Claude 会：
1. 读取相关代码
2. 检查资源加载逻辑
3. 验证 TEngine 规范
4. 提供具体修复建议
```

#### 使用 claude-mem 查找历史解决方案

```
"我记得上次遇到过类似的 UI 显示问题，是怎么解决的？"

→ 触发 claude-mem:mem-search
→ 查找历史经验
→ 直接应用之前的解决方案
```

### 7. 性能优化建议

#### 资源加载优化

```csharp
// ✅ 批量预加载
private async UniTask PreloadResources()
{
    var tasks = new List<UniTask<AssetHandle>>();
    foreach (var path in iconPaths)
    {
        tasks.Add(GameModule.Resource.LoadAssetAsync<Sprite>(path));
    }
    _handles = (await UniTask.WhenAll(tasks)).ToList();
}

// ❌ 逐个加载
private async UniTask LoadResourcesOneByOne()
{
    foreach (var path in iconPaths)
    {
        await GameModule.Resource.LoadAssetAsync<Sprite>(path);
    }
}
```

#### UI 实例化优化

```csharp
// ✅ 使用对象池
var item = await GameModule.UI.CreateUIWidgetFromPool<UIInventoryItem>(prefabPath);

// ❌ 每次都创建新对象
var item = await GameModule.Resource.LoadGameObjectAsync(prefabPath);
```

### 8. 安全注意事项

#### 防止资源泄漏

```csharp
public class UIInventory : UIWindow
{
    private List<AssetHandle> _handles = new();

    // ✅ 统一管理资源句柄
    protected override async UniTask OnCreate()
    {
        var handle = await GameModule.Resource.LoadAssetAsync<Sprite>(path);
        _handles.Add(handle);  // 保存句柄
    }

    // ✅ 确保释放
    protected override void OnDestroy()
    {
        foreach (var handle in _handles)
        {
            GameModule.Resource.UnloadAsset(handle);
        }
        _handles.Clear();
        base.OnDestroy();
    }
}
```

#### 热更边界检查

```bash
# 在提交前检查热更边界
"帮我检查一下 GameScripts/ 目录下的代码是否放在了正确的热更目录"

→ Claude 会扫描代码结构
→ 指出不符合规范的文件
→ 提供修正建议
```

---

## 工具链总览

### 完整技术栈

```mermaid
graph TB
    subgraph "开发环境"
        Claude[Claude Code<br/>AI 编程助手]
        ccswitch[ccswitch<br/>API 管理/反代]
        mem[claude-mem<br/>长期记忆库]
    end

    subgraph "工作流工具"
        openspec[openspec<br/>规范驱动开发]
        skills[TEngine Skills<br/>tengine-dev]
        unity[unity-skills<br/>Unity 自动化]
    end

    subgraph "TEngine 框架"
        hybrid[HybridCLR<br/>热更新]
        yoo[YooAsset<br/>资源管理]
        uni[UniTask<br/>异步编程]
        luban[Luban<br/>配置表]
    end

    subgraph "Unity 引擎"
        editor[Unity Editor]
        build[构建系统]
    end

    Claude --> ccswitch
    Claude --> mem
    Claude --> openspec
    Claude --> skills
    Claude --> unity

    openspec --> skills
    skills --> hybrid
    skills --> yoo
    skills --> uni
    skills --> luban

    unity --> editor
    skills --> editor

    hybrid --> build
    yoo --> build

    style Claude fill:#ffe1f0
    style mem fill:#e1ffe1
    style openspec fill:#e1f0ff
    style skills fill:#fff4e1
    style unity fill:#f0e1ff
```

### 工具职责说明

| 工具 | 职责 | 使用场景 |
|------|------|----------|
| **Claude Code** | AI 编程助手核心 | 所有开发场景 |
| **ccswitch** | API 管理和反代 | 密钥管理、访问加速 |
| **claude-mem** | 长期记忆库 | 知识积累、历史查询 |
| **openspec** | 规范驱动开发 | 需求管理、文档生成 |
| **tengine-dev** | TEngine 开发指导 | 代码开发、规范检查 |
| **unity-skills** | Unity 自动化 | Editor 操作、资源管理 |

### 数据流向

```mermaid
sequenceDiagram
    participant User as 开发者
    participant Claude as Claude Code
    participant Mem as claude-mem
    participant Spec as openspec
    participant Skills as tengine-dev
    participant Unity as Unity Editor

    User->>Claude: 提出需求
    Claude->>Mem: 搜索历史经验
    Mem-->>Claude: 返回相关记忆

    Claude->>Spec: 创建提案
    Spec-->>Claude: 生成 artifacts

    User->>Claude: 确认开始开发
    Claude->>Spec: 加载上下文
    Claude->>Skills: 触发 TEngine 技能

    Skills->>Skills: 生成符合规范的代码
    Skills->>Unity: 调用 unity-skills
    Unity-->>Skills: 完成 Editor 操作

    Claude->>Mem: 保存新经验
    Claude-->>User: 完成开发
```

---

## 相关文档

- [openspec 官方文档](https://github.com/openspec/openspec)
- [TEngine 框架文档](Books/0-介绍.md)
- [tengine-dev 技能参考](skills/tengine-dev/references/)
- [claude-mem 插件](https://github.com/fission-ai/claude-mem)
- [Unity-MCP 指南](skills/tengine-dev/references/unity-mcp-guide.md)

---

## 常见问题

### Q: 为什么需要 ccswitch？

A: ccswitch 提供以下核心功能：
- **多密钥管理**: 自动在多个 API 密钥间切换，提高可用性
- **反向代理**: 在网络受限环境下加速访问
- **负载均衡**: 避免单个密钥的请求限制

### Q: claude-mem 的记忆会占用很多空间吗？

A: 不会。claude-mem 使用向量数据库高效存储：
- 只保存关键信息和经验
- 自动去重和压缩
- 可配置保留策略

### Q: 不使用 openspec 可以吗？

A: 可以，但强烈推荐使用：
- **小需求**: 可以直接让 Claude 编写代码
- **中等需求**: 建议使用 openspec 记录设计
- **大型需求**: 必须使用 openspec 确保需求清晰

### Q: tengine-dev 技能如何触发？

A: 自动触发，无需手动调用：
- 描述中包含 TEngine 关键词即可
- 如：UIWindow、GameModule、YooAsset、UniTask 等
- Claude 会自动加载相应的开发指导

### Q: 如何验证环境配置正确？

A: 检查清单：
```bash
# 1. 检查 openspec
openspec --version

# 2. 检查 ccswitch
ccswitch list

# 3. 在 Claude Code 中测试
"请帮我创建一个简单的 UIWindow"
→ 应该自动触发 tengine-dev 技能

# 4. 测试 claude-mem
"上次我们是怎么处理资源加载的？"
→ 应该能搜索到历史记忆
```

---

## 进阶使用

### 自定义 openspec 模板

可以在项目中创建自定义模板：

```bash
# 在 .openspec/templates/ 目录下创建模板
mkdir -p .openspec/templates/ui-feature

# 编辑模板文件
# proposal-template.md
# design-template.md
# spec-template.md
```

### 扩展 tengine-dev 技能

在 `skills/tengine-dev/references/` 中添加自定义参考文档：

```bash
# 添加项目特定的最佳实践
touch skills/tengine-dev/references/custom-patterns.md
```

### 配置 unity-skills 脚本模板

在 Unity Editor 中配置自定义脚本模板，unity-skills 会自动使用。

---

*如有问题，请提交 Issue 或联系维护者。*
