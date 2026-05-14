---
description: 同步文档与代码，修正不一致内容（自动加载 wiki-synchelper skill）
---

加载 `wiki-synchelper` skill，对当前改动做文档对齐。

**输入**：`$ARGUMENTS`（可选，指定要同步的模块或子系统，例如：`/wiki-sync BattleCore 技能系统`；留空则根据对话上下文自动判断）

**执行步骤**

1. 宣告任务：输出"开始文档同步：$ARGUMENTS"（若无参数则输出"根据上下文自动判断同步范围"）
2. 加载 `wiki-synchelper` skill（使用 Skill 工具，name: "wiki-synchelper"）
3. 按 skill 阶段一执行代码核验：
   - 确定核验范围（来自 $ARGUMENTS 或对话上下文中最近的代码改动）
   - 使用 Glob/Read/Grep 核验涉及的代码文件
4. 按 skill 阶段二扫描现有文档并比对
5. 按 skill 阶段三执行文档操作（修正/补充/删除）
6. 按 skill 阶段四更新关联文档及 `.repowiki/README.md` 导航
7. 输出标准文档更新报告

**完成后必须输出**：

```
文档同步完成。
如果同步结果不符合预期，可以使用 /undo 撤销本次所有文件改动并重试。
```
