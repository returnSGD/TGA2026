# 控件映射表：HTML data-u-type → Unity UGUI 组件

烘焙器根据 JSON 中的 `type` 字段在 Unity 中生成对应的 UGUI 组件结构。

## 映射总表

| data-u-type | Unity 生成结构 | 支持的属性 |
|-------------|---------------|-----------|
| `div` | GameObject + **Image** (Raycast 自动剔除) | 尺寸、坐标、背景色 |
| `image` | GameObject + **Image** | 尺寸、坐标、背景色 |
| `text` | GameObject + **TextMeshProUGUI** (或 Legacy Text) | fontSize、color、textAlign |
| `button` | GameObject + **Image** + **Button** + 子 Text | 背景色、内部文本样式 |
| `input` | GameObject + **Image** + **TMP_InputField** + 子层级 | placeholder、文本样式 |
| `scroll` | GameObject + **Image** + **ScrollRect** + Viewport(Mask) + Content | 滚动方向(dir)、子节点挂到 Content |
| `toggle` | GameObject + **Toggle** + Background + Checkmark + Label | isChecked、文本 |
| `slider` | GameObject + **Slider** + Background + Fill + Handle | value (0~1) |
| `dropdown` | GameObject + **Image** + **TMP_Dropdown** + 完整子层级 | options 列表 |

## 复杂控件内部结构

### Button
```
Button (Image + Button)
  └── Text (TextMeshProUGUI)
```

### Input
```
InputField (Image + TMP_InputField)
  └── Text Area (RectMask2D)
        ├── Placeholder (TextMeshProUGUI)
        └── Text (TextMeshProUGUI)
```

### Scroll
```
ScrollRect (Image + ScrollRect)
  └── Viewport (RectMask2D)
        └── Content (子节点挂载于此)
```

### Toggle
```
Toggle (Toggle)
  ├── Background (Image)
  │     └── Checkmark (Image)
  └── Label (TextMeshProUGUI)
```

### Slider
```
Slider (Slider)
  ├── Background (Image)
  ├── Fill Area
  │     └── Fill (Image)
  └── Handle Slide Area
        └── Handle (Image)
```

### Dropdown
```
Dropdown (Image + TMP_Dropdown)
  ├── Label (TextMeshProUGUI)
  ├── Arrow (Image)
  └── Template (Image + ScrollRect, 默认隐藏)
        └── Viewport (Mask)
              └── Content
                    └── Item (Toggle)
                          ├── Item Background (Image)
                          ├── Item Checkmark (Image)
                          └── Item Label (TextMeshProUGUI)
```

## 文本组件选择

Unity 烘焙器支持两种文本组件，在 Baker 窗口中切换：
- **TMP 模式**（默认）：使用 `TextMeshProUGUI`，推荐
- **Legacy 模式**：使用 `UnityEngine.UI.Text`，勾选"使用旧版 Text"

## RectTransform 定位规则

所有节点统一使用：
- `anchorMin = anchorMax = (0, 1)` — 左上角锚点
- `pivot = (0, 1)` — 左上角轴心
- `anchoredPosition = (localX, -localY)` — Y 轴取反适配 Unity 坐标系
