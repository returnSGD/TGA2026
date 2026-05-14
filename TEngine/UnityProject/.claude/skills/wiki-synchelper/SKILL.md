---
name: wiki-synchelper
description: Wiki同步助手（Wiki SyncHelper）——用于"项目实现内容"与"开发Wiki文档"之间的双向同步。服务对象为AI开发流程，确保AI可基于Wiki快速理解项目现状并按规范继续开发。触发场景：(1) 用户要求扫描/比对/同步/报告项目与Wiki的差异；(2) 代码实现已更新但Wiki文档未跟进；(3) Wiki文档需要反向修正代码结构；(4) 用户说"同步Wiki"、"更新文档"、"Wiki和代码不一致"、"wiki-synchelper"、"扫描文档差异"，或调用 init/scan/diff/sync/report 命令。本技能完全通用，不含任何项目特定路径，通过 wiki-sync.yaml 适配各项目。
---

# Wiki同步助手（wiki-synchelper）

## 设计原则

**技能本身完全通用**，不包含任何项目特定路径或映射。所有项目配置存储于项目根目录的 `wiki-sync.yaml`。多项目复用只需每个项目维护自己的配置文件。

**约定**：Wiki 文档根目录默认为 `repowiki/`（`init` 命令自动识别此目录）。

两个同步方向：
- **方向A（项目→Wiki）**：代码变更后，更新 Wiki 反映最新实现
- **方向B（Wiki→项目）**：Wiki 规范更新后，生成代码合规性检查待办清单

---

## 配置文件发现顺序

执行任何命令前，按以下顺序查找配置（找到即停止）：

1. 命令行参数：`--config <path>`
2. 当前工作目录：`./wiki-sync.yaml`
3. 项目根目录（向上查找含 `.git` 的目录）：`<git-root>/wiki-sync.yaml`
4. `.claude/` 目录：`.claude/wiki-sync.yaml`

未找到时：提示运行 `init` 命令生成配置，然后中止。

---

## 命令接口

### `init` — 初始化配置

**功能**：扫描项目结构，自动生成 `wiki-sync.yaml` 配置模板。

**执行步骤**：
1. 检测项目根目录（寻找 `.git`）
2. 自动发现 Wiki 目录（优先查找 `repowiki/`，其次 `wiki/`、`docs/`）
3. 列出发现的目录，**询问用户确认 source_paths 和 wiki_root**
4. 生成 `wiki-sync.yaml`（`write_enabled: false`，只读模式）

---

### `scan` — 扫描

**功能**：只读扫描，枚举项目模块与 Wiki 文档的对应关系，不修改任何文件。

**输出示例**：
```
[SCAN] 扫描结果
  配置：<git-root>/wiki-sync.yaml
  项目模块数：12 | Wiki文档数：10 | 已覆盖：9
  未覆盖（项目有/Wiki无）：2 → [ModuleA, ModuleB]
  孤立Wiki（项目已删除/未映射）：1 → [repowiki/zh/content/OldSystem.md]
```

**执行步骤**：
1. 读取并验证配置（路径不存在则中止报错）
2. 遍历 `source_paths`，按 `ignore.source` 过滤，建立模块列表
3. 遍历 `wiki_root`，按 `ignore.wiki` 过滤，建立文档列表
4. 按 `mappings` 做交叉比对，输出覆盖状态表

---

### `diff` — 比对

**功能**：对已匹配的模块/文档对，检测内容差异（只读）。

**参数**：`--target <模块名>` 限定单个模块；`--dir A|B|both`（默认 both）

**输出示例**：
```
[DIFF] ModuleA/SkillSystem ↔ repowiki/zh/content/.../技能系统.md
  ▸ [新增] 枚举值 SkillPhase.Cast（Wiki未记录）
  ▸ [重命名] SkillCooldown → CooldownTick（代码已更新，Wiki未跟进）
  ▸ [孤立] Wiki描述的 SkillPrecast() 在代码中已删除
  建议：sync --dir A --target ModuleA/SkillSystem
```

**执行步骤**：
1. 执行 `scan`，获取已匹配对
2. 对每对：读取代码结构（类名、公共方法、枚举）+ 读取 Wiki 内容
3. 对比并标记差异类型：`新增` / `删除` / `重命名` / `语义变更`

---

### `sync` — 同步

**功能**：根据差异执行实际写入，**需显式开启写入模式**。

**开启写入**：配置 `write_enabled: true` 或命令行传入 `--write`。

**同步方向**：
- `--dir A`：项目→Wiki（更新 Wiki 文档）
- `--dir B`：Wiki→项目（生成 `WIKI_SYNC_TODO.md`，**不自动改代码**）
- `--dir both`：双向，冲突按策略处理

**冲突策略**（`conflict_strategy`）：

| 值 | 行为 |
|----|------|
| `ask`（默认）| 每个冲突项暂停，等待用户决策 |
| `prefer_code` | 以代码为准，覆盖 Wiki |
| `prefer_wiki` | 以 Wiki 为准，生成代码修改建议 |
| `skip` | 跳过冲突项，记录到报告 |

**执行步骤**：
1. 检查 `write_enabled`，为 false 时拒绝并提示
2. 执行 `diff`，获取差异列表
3. 写入前备份目标文件
4. 检测敏感信息 → 脱敏 → 写入
5. 输出操作日志

---

### `report` — 报告

**功能**：生成完整同步状态报告（Markdown），可输出到文件。

**参数**：`--output <path>`（默认输出到控制台）

**输出内容**：覆盖率统计、差异概览、建议操作清单

---

## 配置文件规范

**文件名**：`wiki-sync.yaml`（项目根目录，不提交到技能目录）

```yaml
# wiki-sync.yaml — 项目特定配置，不属于技能本身
version: "1.0"

source_paths:
  - path: "src"          # 相对于项目根目录，按实际填写
    label: "Source"

wiki_root: "repowiki"    # Wiki 文档根目录，默认约定为 repowiki/

mappings:
  - source_pattern: "src/ModuleA/**"
    wiki_path: "zh/content/ModuleA"

ignore:
  source: ["**/*.meta", "**/bin/**", "**/obj/**"]
  wiki:   ["**/草稿/**", "**/Archive/**", "**/_*"]

conflict_strategy: ask   # ask | prefer_code | prefer_wiki | skip

write_enabled: false     # 必须显式改为 true 才允许写入

backup:
  enabled: true
  directory: ".wiki-sync-backups"

sensitive_patterns:
  - pattern: "(api[_-]?key|secret|password|token|pwd)\\s*[:=]\\s*\\S+"
    replacement: "<REDACTED>"

report:
  default_output: "wiki-sync-report.md"
  include_diff_detail: true
  include_suggestions: true
```

> 完整配置字段说明见 [references/config-reference.md](references/config-reference.md)

---

## 典型工作流

**新项目首次使用**：
```
init    → 扫描项目，生成 wiki-sync.yaml（需确认路径）
scan    → 验证配置，查看覆盖率
```

**代码更新后同步 Wiki**：
```
scan                        → 查看哪些模块缺少 Wiki
diff                        → 查看现有 Wiki 的过时内容
sync --dir A --write        → 更新 Wiki（需先将 write_enabled 改为 true）
report                      → 生成变更报告
```

**Wiki 规范更新后检查代码合规**：
```
diff --dir B                → 检查代码是否符合最新 Wiki 规范
report --output check.md   → 方向B输出 WIKI_SYNC_TODO.md，供开发者手动跟进
```

---

## 异常处理

| 异常 | 行为 |
|------|------|
| 配置文件不存在 | **中止**，提示运行 `init` |
| 路径不存在 | **中止**，列出缺失路径 |
| 无写入权限 | **中止** |
| `write_enabled: false` 时调用 sync | **拒绝**，提示修改配置或传入 `--write` |
| 两侧都有变更（冲突） | 按 `conflict_strategy` 处理 |
| 检测到敏感信息 | 脱敏后继续，日志标注位置 |
| Wiki 文档格式错误 | 警告后跳过，继续处理其余 |

---

## 安全约束（强制）

1. **默认只读**：所有命令默认不修改任何文件
2. **脱敏必须**：敏感信息不得写入 Wiki
3. **备份优先**：写入前强制备份
4. **方向B只读代码**：不自动修改代码，只生成待办清单
5. **错误即停**：任一前置条件不满足，立即中止，不做部分写入
