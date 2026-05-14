# Luban 运行时加载

## Unity + C# + JSON

### 安装 Luban.Runtime

Unity 用户可安装 `com.code-philosophy.luban` 包：

```
https://gitee.com/focus-creative-games/luban_unity.git
```

### 基础加载代码

```csharp
using SimpleJSON;

public class ConfigLoader : MonoBehaviour
{
    private cfg.Tables _tables;

    void Start()
    {
        _tables = new cfg.Tables(Loader);

        // 使用数据
        var item = _tables.TbItem.Get(12);
        Debug.Log(item.Name);
    }

    private static JSONNode Loader(string file)
    {
        var path = Path.Combine(Application.streamingAssetsPath, $"{file}.json");
        return JSON.Parse(File.ReadAllText(path, System.Text.Encoding.UTF8));
    }
}
```

## Unity + C# + Binary

### 基础加载代码

```csharp
using Luban;

public class ConfigLoader : MonoBehaviour
{
    private cfg.Tables _tables;

    void Start()
    {
        _tables = new cfg.Tables(Loader);
    }

    private static ByteBuf Loader(string file)
    {
        var path = Path.Combine(Application.streamingAssetsPath, $"{file}.bytes");
        return new ByteBuf(File.ReadAllBytes(path));
    }
}
```

## 自动判断格式

通过反射检测 `cfg.Tables` 构造函数的 Loader 返回类型，自动适配 JSON 或 Binary：

```csharp
var loader = loaderReturnType == typeof(ByteBuf) ?
    new System.Func<string, ByteBuf>(LoadByteBuf) :
    (System.Delegate)new System.Func<string, JSONNode>(LoadJson);

var tables = (cfg.Tables)tablesCtor.Invoke(new object[] {loader});
```

## 类型映射（TypeMapper）

### 将配置类型映射到外部类型

**仅 C# 代码支持**

### Enum 映射

```xml
<enum name="AudioType">
    <var name="UNKNOWN" value="0"/>
    <mapper target="client" codeTarget="cs-bin">
        <option name="type" value="UnityEngine.AudioType"/>
    </mapper>
</enum>
```

**注意：** 必须确保枚举项的值与映射的枚举类型的枚举项的值完全一致。

### Bean 映射

```xml
<bean name="vector3" valueType="1" sep=",">
    <var name="x" type="float"/>
    <var name="y" type="float"/>
    <var name="z" type="float"/>
    <mapper target="client" codeTarget="cs-bin">
        <option name="type" value="UnityEngine.Vector3"/>
        <option name="constructor" value="ExternalTypeUtil.NewVector3"/>
    </mapper>
</bean>
```

Bean 需额外提供 `constructor` 指定转换函数。

### 匹配规则

通过命令行 `-t $target` 和 `-c $codeTarget` 匹配 mapper 的对应属性。

```bash
-t client -c cs-bin
```

`target`/`codeTarget` 支持多值：

```xml
<mapper target="client,server" codeTarget="cs-bin">
```

## 代码风格

### 命名风格

| 风格 | 说明 | 示例 |
|------|------|------|
| `none` | 保持原样 | `aa_bb_cc` → `aa_bb_cc` |
| `camel` | 小驼峰 | `aa_bb_cc` → `aaBbCc` |
| `pascal` | 大驼峰 | `aa_bb_cc` → `AaBbCc` |
| `upper` | 全大写 | `aa_bb_cc` → `AA_BB_CC` |
| `snake` | 下划线 | `aa_bb_cc` → `aa_bb_cc` |

### 命名位置

- `namespace` — 命名空间
- `type` — 类型名（enum, bean, table, manager）
- `method` — 函数名
- `property` — 属性名
- `field` — 字段名
- `enumItem` — 枚举项名

### 默认风格（按语言）

| 语言 | namespace | type | method | property | field | enumItem |
|------|-----------|------|--------|----------|-------|----------|
| C# | pascal | pascal | pascal | pascal | camel | none |
| Java | pascal | pascal | pascal | camel | camel | none |
| Go | snake | pascal | camel | camel | pascal | none |
| TypeScript | pascal | pascal | camel | camel | camel | none |

### 命令行设置

```bash
-x codeStyle=your_style
-x namingConvention.{codeTarget}.{location}=style
```

## 本地化系统

### 配置方法

通过命令行参数启用：

```bash
-x l10n.provider=default
-x l10n.textFile.path=*@path/to/texts.json
-x l10n.textFile.keyFieldName=key
-x l10n.textFile.languageFieldName=zh
-x l10n.convertTextKeyToValue=1
```

### text 类型

`text` 是特殊的语法糖，等价 `string#text=1`，语义上表示本地化字符串的 key。

### 时间本地化

datetime 按目标时区转 UTC 秒数：

```bash
-t "Asia/Shanghai"
```
