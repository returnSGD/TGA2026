#!/usr/bin/env python3
"""
HTML to JSON UI Baker — Python 版坐标烘焙器

将符合 UI-DSL 规范的 HTML 文件烘焙为 Unity UGUI 可消费的 JSON 坐标数据。
核心逻辑移植自 HtmlToJson/HTML 转 JSON 坐标烘焙器.html，使用 Playwright 浏览器渲染精确提取坐标。

依赖安装：pip install playwright && playwright install chromium

用法：
  python bake_html_to_json.py input.html -o output.json
  python bake_html_to_json.py input.html --width 1920 --height 1080
  python bake_html_to_json.py input.html --stdout   # 输出到标准输出
"""

import argparse
import json
import sys
import os

def bake_html_to_json(html_content: str, width: int = 1920, height: int = 1080) -> dict:
    """
    使用 Playwright 渲染 HTML 并提取 UGUI JSON 坐标数据。
    逻辑与原始 JS 烘焙器完全一致。
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("错误: 需要安装 playwright。运行: pip install playwright && playwright install chromium", file=sys.stderr)
        sys.exit(1)

    # 构建完整 HTML 页面（内嵌烘焙脚本）
    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
body {{ margin: 0; padding: 0; }}
#canvas-sandbox {{ position: relative; width: {width}px; height: {height}px; }}
#canvas-sandbox * {{ box-sizing: border-box !important; }}
#canvas-sandbox [data-u-type] {{ min-width: 0; min-height: 0; }}
</style>
</head><body>
<div id="canvas-sandbox">{html_content}</div>
<script>
function rgb2hex(rgb) {{
    if (!rgb || rgb === 'rgba(0, 0, 0, 0)' || rgb === 'transparent') return '#FFFFFF00';
    const match = rgb.match(/^rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)(?:,\\s*([\\d.]+))?\\)$/);
    if (!match) return '#FFFFFF';
    const r = ("0" + parseInt(match[1], 10).toString(16)).slice(-2);
    const g = ("0" + parseInt(match[2], 10).toString(16)).slice(-2);
    const b = ("0" + parseInt(match[3], 10).toString(16)).slice(-2);
    const a = match[4] ? ("0" + Math.round(parseFloat(match[4]) * 255).toString(16)).slice(-2) : "ff";
    return `#${{r}}${{g}}${{b}}${{a === 'ff' ? '' : a}}`;
}}

function traverseAndBake(element, rootRect) {{
    const uType = element.getAttribute('data-u-type');
    const uName = element.getAttribute('data-u-name');
    let nodeData = null;

    if (uType && uName) {{
        const rect = element.getBoundingClientRect();
        const style = window.getComputedStyle(element);
        const relativeX = rect.left - rootRect.left;
        const relativeY = rect.top - rootRect.top;
        const realWidth = rect.width;
        const realHeight = rect.height;

        let textContent = element.innerText || "";
        if (element.tagName.toLowerCase() === 'input') {{
            textContent = element.value || element.placeholder || "";
        }}

        let fontSize = 14;
        if (style.fontSize) fontSize = parseFloat(style.fontSize);
        let textAlign = style.textAlign || 'center';
        let uDir = element.getAttribute('data-u-dir') || 'v';
        let uValue = parseFloat(element.getAttribute('data-u-value')) || 0.5;
        let uChecked = element.getAttribute('data-u-checked') === 'true';
        let uOptions = [];

        if (uType === 'dropdown' && element.tagName.toLowerCase() === 'select') {{
            const opts = element.querySelectorAll('option');
            opts.forEach(opt => uOptions.push(opt.innerText.trim()));
        }}

        nodeData = {{
            name: uName,
            type: uType,
            dir: uDir,
            value: uValue,
            isChecked: uChecked,
            options: uOptions,
            x: Math.round(relativeX),
            y: Math.round(relativeY),
            width: Math.round(realWidth),
            height: Math.round(realHeight),
            color: rgb2hex(style.backgroundColor),
            fontColor: rgb2hex(style.color),
            fontSize: Math.round(fontSize),
            textAlign: textAlign,
            text: textContent.trim(),
            children: []
        }};
    }}

    const childrenData = [];
    for (let i = 0; i < element.children.length; i++) {{
        if (element.tagName.toLowerCase() === 'select' && element.children[i].tagName.toLowerCase() === 'option') {{
            continue;
        }}
        const childResult = traverseAndBake(element.children[i], rootRect);
        if (childResult) childrenData.push(childResult);
    }}

    if (nodeData) {{
        nodeData.children = childrenData;
        return nodeData;
    }} else if (childrenData.length > 0) {{
        return childrenData.length === 1 ? childrenData[0] : {{
            name: "layoutGroup_" + Math.random().toString(36).substr(2, 5),
            type: "div", dir: "v", value: 0, isChecked: false, options: [],
            x: 0, y: 0, width: 0, height: 0,
            color: "#FFFFFF00", fontColor: "#000000", fontSize: 14, textAlign: "center", text: "", children: childrenData
        }};
    }}
    return null;
}}

// 烘焙入口
const sandbox = document.getElementById('canvas-sandbox');
const rootElement = sandbox.querySelector('[data-u-name]');
if (!rootElement) {{
    throw new Error('未找到包含 data-u-name 的根节点');
}}
const rootRect = rootElement.getBoundingClientRect();
const result = traverseAndBake(rootElement, rootRect);
window.__BAKE_RESULT__ = result;
</script>
</body></html>"""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width + 100, "height": height + 100})
        page.set_content(full_html, wait_until="networkidle")

        # 等待渲染完成
        page.wait_for_timeout(500)

        # 提取烘焙结果
        result = page.evaluate("window.__BAKE_RESULT__")

        browser.close()

    if result is None:
        raise ValueError("烘焙失败: 未找到有效的 UI-DSL 节点，请检查 HTML 是否包含 data-u-name 和 data-u-type 属性")

    return result


def main():
    parser = argparse.ArgumentParser(description="HTML to JSON UI Baker — 将 UI-DSL HTML 烘焙为 UGUI JSON")
    parser.add_argument("input", help="输入 HTML 文件路径")
    parser.add_argument("-o", "--output", help="输出 JSON 文件路径（默认：同名 .json）")
    parser.add_argument("-w", "--width", type=int, default=1920, help="画布宽度（默认 1920）")
    parser.add_argument("-H", "--height", type=int, default=1080, help="画布高度（默认 1080）")
    parser.add_argument("--stdout", action="store_true", help="输出到标准输出而非文件")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        html_content = f.read()

    result = bake_html_to_json(html_content, args.width, args.height)
    json_str = json.dumps(result, ensure_ascii=False, indent=2)

    if args.stdout:
        print(json_str)
    else:
        output_path = args.output or os.path.splitext(args.input)[0] + ".json"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"烘焙完成: {output_path}")


if __name__ == "__main__":
    main()
