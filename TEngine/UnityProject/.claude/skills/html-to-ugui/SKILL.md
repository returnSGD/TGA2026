---
name: html-to-ugui
description: "HTML原型转Unity UGUI界面生成管线。通过AI生成符合UI-DSL规范的HTML，烘焙为JSON坐标数据，再导入Unity自动生成UGUI节点树。触发场景：(1) 需要快速生成Unity UGUI界面原型 (2) 用自然语言描述UI需求并自动生成 (3) 创建UIWindow/面板的初始布局 (4) 批量生成表单、设置、列表等标准界面"
---

# HTML to UGUI 原型生成管线
你是一个专业的 UI 原型开发专家。你需要生成要求兼顾美观性、可用性与实现可行性 规范可直接用于开发的游戏的 HTML 代码 UI设计方案，用于导入 Unity 引擎全自动生成 UGUI 界面。

将自然语言 UI 需求 → UI-DSL HTML → JSON 坐标数据 → Unity UGUI 节点树。

## 工作流（三步）

### Step 1: 生成 UI-DSL HTML

读取 [references/ui-dsl-spec.md](references/ui-dsl-spec.md) 获取完整规范，然后根据用户需求生成 HTML。

核心规则：
- 唯一根节点：`data-u-type="div"` + `data-u-name="rootName"` + 明确 `width/height`
- 所有节点必须有 `data-u-name`（小驼峰）和 `data-u-type`（8种之一）
- 仅允许：`div` `image` `text` `button` `input` `scroll` `toggle` `slider` `dropdown`
- 使用 CSS Flexbox 布局，绝对不可超出根节点尺寸

### Step 2: 烘焙 HTML → JSON

运行烘焙脚本将 HTML 转换为 Unity 可消费的 JSON 坐标数据：

```bash
python scripts/bake_html_to_json.py input.html -o output.json -w 1920 -h 1080
```

也可直接在 HTML 输出中内嵌 JSON——脚本会输出到 stdout。

JSON 结构参见 [references/json-schema.md](references/json-schema.md)。

### Step 3: 导入 Unity

将 JSON 粘贴到 Unity Editor 的 HTML to UGUI Baker 窗口：
1. 打开 `Tools > UI Architecture > HTML to UGUI Baker`
2. 选择目标 Canvas，切换到"直接粘贴 JSON 字符"模式
3. 粘贴 JSON，点击"执行烘焙生成"

控件映射参见 [references/control-mapping.md](references/control-mapping.md)。

## 快速参考

### 分辨率预设

| 预设 | 尺寸 |
|------|------|
| PC 横屏 | 1920x1080 |
| Mobile 竖屏 | 750x1330 |
| Pad 横屏 | 2048x1536 |

### 控件速查

| data-u-type | 用途 | 特殊属性 |
|-------------|------|---------|
| `div` | 容器/背景 | - |
| `image` | 图片占位 | - |
| `text` | 文本 | `text-align` |
| `button` | 按钮 | - |
| `input` | 输入框 | `placeholder` |
| `scroll` | 滚动列表 | `data-u-dir="v/h"` |
| `toggle` | 开关 | `data-u-checked="true"` |
| `slider` | 滑动条 | `data-u-value="0.5"` |
| `dropdown` | 下拉菜单 | `<option>` 子标签 |

## 典型使用场景

**场景1：快速生成设置界面**
> 用户："帮我做一个游戏设置界面，包含音量滑条、全屏开关、画质下拉框"

→ 读取 ui-dsl-spec.md → 生成 HTML → 运行 bake 脚本 → 输出 JSON → 用户粘贴到 Unity

**场景2：生成登录窗口**
> 用户："做一个登录界面，有账号输入、密码输入、登录按钮、记住密码开关"

→ 同上流程，注意 `input` 的 `placeholder` 属性和 `toggle` 的 `data-u-checked`
