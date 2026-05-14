# JSON 输出格式（UIDataNode）

烘焙器从 HTML 提取坐标后输出如下 JSON 结构，供 Unity Editor `HtmlToUGUIBaker.cs` 消费。

## 节点结构

```json
{
  "name": "string",          // data-u-name 值，小驼峰
  "type": "string",          // data-u-type 值：div/image/text/button/input/scroll/toggle/slider/dropdown
  "dir": "string",           // 滚动方向："v"(垂直) 或 "h"(水平)，非 scroll 类型为 "v"
  "value": 0.5,              // Slider 默认进度 0.0~1.0，非 slider 类型为 0.5
  "isChecked": false,        // Toggle 默认状态，非 toggle 类型为 false
  "options": ["选项1","选项2"], // Dropdown 选项列表，非 dropdown 类型为 []
  "x": 0,                    // 相对根节点的 X 坐标 (px)
  "y": 0,                    // 相对根节点的 Y 坐标 (px)
  "width": 1920,             // 节点宽度 (px)
  "height": 1080,            // 节点高度 (px)
  "color": "#2C3E50",        // 背景色 (#RRGGBB 或 #RRGGBBAA)，透明为 #FFFFFF00
  "fontColor": "#FFFFFF",    // 字体颜色
  "fontSize": 24,            // 字体大小 (px)
  "textAlign": "center",     // 文本对齐：left/right/center
  "text": "显示文本",         // 节点内文本内容
  "children": []             // 子节点数组，结构相同
}
```

## 坐标系说明

- 原点在根节点左上角
- X 向右递增，Y 向下递增（与 Unity UI 的 `anchoredPosition` 一致）
- Unity 端使用 `anchorMin/Max = (0,1)` + `pivot = (0,1)` 定位
- `anchoredPosition = (localX, -localY)`，其中 `localX/Y` = 子节点绝对坐标 - 父节点绝对坐标

## 完整示例

```json
{
  "name": "settingsWindow",
  "type": "div",
  "dir": "v",
  "value": 0.5,
  "isChecked": false,
  "options": [],
  "x": 0,
  "y": 0,
  "width": 1920,
  "height": 1080,
  "color": "#2C3E50",
  "fontColor": "#D4D4D4",
  "fontSize": 16,
  "textAlign": "start",
  "text": "",
  "children": [
    {
      "name": "titleTxt",
      "type": "text",
      "dir": "v",
      "value": 0.5,
      "isChecked": false,
      "options": [],
      "x": 710,
      "y": 40,
      "width": 500,
      "height": 60,
      "color": "#FFFFFF00",
      "fontColor": "#FFFFFF",
      "fontSize": 48,
      "textAlign": "center",
      "text": "系统设置",
      "children": []
    },
    {
      "name": "volumeSlider",
      "type": "slider",
      "dir": "v",
      "value": 0.8,
      "isChecked": false,
      "options": [],
      "x": 40,
      "y": 140,
      "width": 400,
      "height": 30,
      "color": "#7F8C8D",
      "fontColor": "#000000",
      "fontSize": 14,
      "textAlign": "center",
      "text": "",
      "children": []
    }
  ]
}
```
