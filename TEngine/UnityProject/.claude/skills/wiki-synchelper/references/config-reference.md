# 配置字段完整说明

## 顶层字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `version` | string | 是 | — | 配置版本，当前为 `"1.0"` |
| `source_paths` | list | 是 | — | 项目源码目录列表 |
| `wiki_root` | string | 是 | — | Wiki文档根目录（相对于项目根目录，通常为 `repowiki`） |
| `mappings` | list | 否 | 空 | 路径映射规则；为空时对 source_paths 与 wiki_root 做平铺匹配 |
| `ignore` | object | 否 | — | 忽略规则，含 `source` / `wiki` 两个 glob 列表 |
| `conflict_strategy` | string | 否 | `ask` | 冲突处理策略 |
| `write_enabled` | bool | 否 | `false` | 是否允许写入，必须显式设为 true |
| `backup` | object | 否 | — | 备份设置 |
| `sensitive_patterns` | list | 否 | — | 敏感信息脱敏规则 |
| `report` | object | 否 | — | 报告输出设置 |

---

## `source_paths`

```yaml
source_paths:
  - path: "src"        # 相对于项目根目录
    label: "Source"    # 可选，用于日志显示；默认取路径末段
```

---

## `wiki_root`

指向 Wiki 文档根目录。本技能约定优先查找 `repowiki/`，`init` 命令会自动发现并填入此字段。

```yaml
wiki_root: "repowiki"
```

---

## `mappings`

将 source_paths 下的子路径映射到 wiki_root 下的子目录。支持 glob 模式。

```yaml
mappings:
  - source_pattern: "src/ModuleA/**"   # glob，匹配 source_paths 下的子路径
    wiki_path: "zh/content/ModuleA"    # 对应 wiki_root 下的目录
```

**不填 mappings 时**：按文件名相似度做模糊匹配，结果会在 `scan` 输出中标注置信度。

**`source_pattern` 语法**：标准 glob（`*`、`**`、`?`）。

---

## `ignore`

```yaml
ignore:
  source:        # 扫描源码时忽略
    - "**/*.meta"
    - "**/bin/**"
    - "**/obj/**"
  wiki:          # 扫描 Wiki 时忽略
    - "**/草稿/**"
    - "**/Archive/**"
    - "**/_*"
```

---

## `conflict_strategy`

| 值 | 行为 |
|----|------|
| `ask`（默认）| 每个冲突项暂停，等待用户交互决策 |
| `prefer_code` | 以代码为准，覆盖 Wiki 中冲突内容 |
| `prefer_wiki` | 以 Wiki 为准，生成代码修改建议（不自动改代码） |
| `skip` | 跳过所有冲突项，在报告中标记 |

---

## `backup`

```yaml
backup:
  enabled: true                    # 默认 true
  directory: ".wiki-sync-backups"  # 相对于项目根目录
  max_versions: 5                  # 每文件保留最多 N 个备份，不填则不限
```

---

## `sensitive_patterns`

写入 Wiki 前自动检测并替换敏感内容（Python `re` 模块语法）：

```yaml
sensitive_patterns:
  - pattern: "(api[_-]?key|secret|password|token|pwd)\\s*[:=]\\s*\\S+"
    replacement: "<REDACTED>"
  - pattern: "[A-Za-z0-9+/]{40,}={0,2}"   # Base64 密钥
    replacement: "<REDACTED_BASE64>"
```

---

## `report`

```yaml
report:
  default_output: "wiki-sync-report.md"   # 默认报告路径
  include_diff_detail: true               # 是否包含详细差异
  include_suggestions: true              # 是否包含修复建议
```
