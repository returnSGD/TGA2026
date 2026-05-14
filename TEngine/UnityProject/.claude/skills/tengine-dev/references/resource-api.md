# 资源加载核心 API

> **适用场景**：SetSprite/LoadGameObjectAsync/LoadAssetAsync 加载、UnloadAsset/UnloadUnusedAssets 卸载、热更下载 API | **关联文档**：[resource-patterns.md](resource-patterns.md)（生命周期模式）、[ui-lifecycle.md](ui-lifecycle.md)（窗口内资源释放时机）

## 核心原则

1. **禁止 `Resources.Load()`**：所有资源通过 YooAsset 加载，放在 `Assets/AssetRaw/` 下
2. **Sprite 用 `SetSprite` 扩展方法**：内置缓存池管理，无需手动释放
3. **GameObject 用 `LoadGameObjectAsync`**：自动管理引用计数，Destroy 时自动卸载
4. **其他 Asset 加载/释放必须配对**：`LoadAssetAsync<T>` → 用完后 → `UnloadAsset`
5. **异步优先**：禁止同步加载大资源

禁止模式详见 [naming-rules.md](naming-rules.md#禁止的代码模式)。

---

## 一、核心 API

### 资源寻址

YooAsset 通过 **location**（文件名，不含路径和扩展名）寻址：

```
Assets/AssetRaw/UI/Prefabs/BattleMainUI.prefab  →  location：BattleMainUI
Assets/AssetRaw/Audios/BGM/MainTheme.mp3        →  location：MainTheme
```

同名文件可使用相对路径去重：`UI/BattleMainUI`

### 加载方式选择

| 资源类型 | 推荐 API | 需要手动释放 |
|---------|---------|-------------|
| Sprite / 图集子图 | `SetSprite` / `SetSubSprite` | 否 |
| 需实例化的 Prefab | `LoadGameObjectAsync` | 否（Destroy 自动）|
| TextAsset / SO 等 | `LoadAssetAsync<T>` | **是** |

### SetSprite 扩展方法（4 个签名）

```csharp
// 1. Image — 基础
void SetSprite(this Image image, string location,
    bool setNativeSize = false, Action<Image> callback = null,
    CancellationToken cancellationToken = default)

// 2. SpriteRenderer
void SetSprite(this SpriteRenderer spriteRenderer, string location,
    Action<SpriteRenderer> callback = null,
    CancellationToken cancellationToken = default)

// 3. Image 图集子图（无回调重载）
void SetSubSprite(this Image image, string location, string spriteName,
    bool setNativeSize = false, CancellationToken cancellationToken = default)

// 4. SpriteRenderer 图集子图（无回调重载）
void SetSubSprite(this SpriteRenderer spriteRenderer, string location,
    string spriteName, CancellationToken cancellationToken = default)
```

> **要点**：SetSprite 的 callback 类型是 `Action<Image>` / `Action<SpriteRenderer>`（不是 `Action<Sprite>`）。SetSubSprite **没有** Action 回调重载。

使用示例：

```csharp
_imgIcon.SetSprite("item_icon_001");                                    // 基础
_imgIcon.SetSprite("item_icon_001", setNativeSize: true);               // 自适应尺寸
_imgIcon.SetSprite("item_icon_001", cancellationToken: _cts.Token);    // 取消支持
_imgIcon.SetSprite("item_icon_001", callback: img => { /* 加载完成 */ }); // 完成回调（Action<Image>）
_spriteRenderer.SetSprite("hero_sprite");                               // SpriteRenderer
_spriteRenderer.SetSprite("hero_sprite", callback: sr => { /* 加载完成 */ }); // Action<SpriteRenderer>
_imgIcon.SetSubSprite("ItemAtlas", "item_sword_01");                    // 图集子图
_imgIcon.SetSubSprite("ItemAtlas", "item_sword_01", setNativeSize: true); // 子图 + 自适应
_spriteRenderer.SetSubSprite("ItemAtlas", "hero_sword");                // SpriteRenderer 图集子图
```

禁止用 `LoadAssetAsync<Sprite>` 加载图片（无缓存池且需手动释放）。

### GameObject 加载

```csharp
// 异步（推荐）
UniTask<GameObject> LoadGameObjectAsync(string location, Transform parent = null,
    CancellationToken cancellationToken = default, string packageName = "")

// 同步（需资源已预加载）
GameObject LoadGameObject(string location, Transform parent = null, string packageName = "")

// 回收：直接 Destroy
Destroy(go);  // 框架自动归还引用计数
```

禁止 `LoadAssetAsync<GameObject>` + `Instantiate` 组合（需手动追踪和 UnloadAsset）。

### Asset 加载与释放

```csharp
// 异步加载
UniTask<T> LoadAssetAsync<T>(string location, CancellationToken cancellationToken = default,
    string packageName = "") where T : UnityEngine.Object

// 异步加载（非泛型）
UniTask<UnityEngine.Object> LoadAssetAsync(string location, Type assetType,
    CancellationToken cancellationToken = default, string packageName = "")

// 同步加载
T LoadAsset<T>(string location, string packageName = "") where T : UnityEngine.Object

// 同步加载（非泛型）
UnityEngine.Object LoadAsset(string location, Type assetType, string packageName = "")

// 回调式异步加载
UniTaskVoid LoadAsset<T>(string location, Action<T> callback, string packageName = "") where T : UnityEngine.Object

// 释放
void UnloadAsset(object asset)
```

### 资源卸载

```csharp
UnloadAsset(asset)                                    // 引用计数-1
UnloadUnusedAssets()                                  // 卸载引用计数为0的资源
ForceUnloadUnusedAssets(performGCCollect: true)       // 强制整理 + 可选 GC
ForceUnloadAllAssets()                                // 退出游戏时
```

### 资源信息查询

```csharp
bool valid = CheckLocationValid(string location, string packageName = "")
HasAssetResult result = HasAsset(string location, string packageName = "")  // Valid/NotExist/AssetOnline/AssetOnDisk
AssetInfo info = GetAssetInfo(string location, string packageName = "")
AssetInfo[] infos = GetAssetInfos(string tag, string packageName = "")      // 按标签
AssetInfo[] infos = GetAssetInfos(string[] tags, string packageName = "")   // 按标签数组
```

### 句柄式加载

返回 `AssetHandle` 的加载方式，适合需要精细控制加载过程的场景：

```csharp
// 同步句柄（泛型 + 非泛型）
AssetHandle LoadAssetSyncHandle<T>(string location, string packageName = "")
AssetHandle LoadAssetSyncHandle(string location, Type assetType, string packageName = "")

// 异步句柄（泛型 + 非泛型）
AssetHandle LoadAssetAsyncHandle<T>(string location, string packageName = "")
AssetHandle LoadAssetAsyncHandle(string location, Type type, string packageName = "")
```

使用示例：

```csharp
using var syncHandle = GameModule.Resource.LoadAssetSyncHandle<Sprite>("icon");
if (syncHandle.IsValid) { var sprite = syncHandle.AssetObject as Sprite; }

using var asyncHandle = GameModule.Resource.LoadAssetAsyncHandle<Sprite>("icon");
await asyncHandle.Task;
if (asyncHandle.IsValid) { var sprite = asyncHandle.AssetObject as Sprite; }
```

### 热更/下载 API

```csharp
// 获取本地资源包版本号
string GetPackageVersion(string customPackageName = "")

// 请求远端版本号
RequestPackageVersionOperation RequestPackageVersionAsync(
    bool appendTimeTicks = false, int timeout = 60, string customPackageName = "")

// 更新资源清单
UpdatePackageManifestOperation UpdatePackageManifestAsync(
    string packageVersion, int timeout = 60, string customPackageName = "")

// 创建差量下载器
ResourceDownloaderOperation CreateResourceDownloader(string customPackageName = "")

// 清理冗余缓存文件
ClearCacheFilesOperation ClearCacheFilesAsync(
    EFileClearMode clearMode = EFileClearMode.ClearUnusedBundleFiles, string customPackageName = "")

// 清理沙盒路径所有缓存
void ClearAllBundleFiles(string customPackageName = "")

// 设置远端资源服务地址（需同时提供默认和备用地址）
void SetRemoteServicesUrl(string defaultHostServer, string fallbackHostServer)
```

---

## 二、使用模式

### 自动管理（无需手动释放）

```csharp
// Sprite：SetSprite 内置缓存池
_imgIcon.SetSprite("item_icon_001");
_imgIcon.SetSubSprite("ItemAtlas", "item_sword_01");

// GameObject：LoadGameObjectAsync 自动引用计数
var go = await GameModule.Resource.LoadGameObjectAsync("HeroPrefab", parent);
Destroy(go);  // 框架自动归还

// 禁止 LoadAssetAsync<Sprite> 或 LoadAssetAsync<GameObject> + Instantiate
```

### 手动管理（必须配对释放）

```csharp
private TextAsset _configData;

protected override async void OnRefresh()
{
    _configData = await GameModule.Resource.LoadAssetAsync<TextAsset>("level_data");
}

protected override void OnDestroy()
{
    if (_configData != null) { GameModule.Resource.UnloadAsset(_configData); _configData = null; }
}
```

### CancellationToken 取消加载

```csharp
private CancellationTokenSource _cts = new();

private async UniTaskVoid LoadAsync()
{
    try { var asset = await GameModule.Resource.LoadAssetAsync<TextAsset>("config", _cts.Token); }
    catch (OperationCanceledException) { /* 正常取消 */ }
}

protected override void OnDestroy() { _cts.Cancel(); _cts.Dispose(); }
```

### 并发与批量加载

```csharp
// 多类型并发
var (config, go, audio) = await UniTask.WhenAll(
    GameModule.Resource.LoadAssetAsync<TextAsset>("hero_config"),
    GameModule.Resource.LoadGameObjectAsync("HeroModel"),
    GameModule.Resource.LoadAssetAsync<AudioClip>("hero_voice")
);

// 同类型批量
var configs = await UniTask.WhenAll(
    locations.Select(loc => GameModule.Resource.LoadAssetAsync<TextAsset>(loc)));
```

### 场景切换资源整理

```csharp
await GameModule.Scene.LoadSceneAsync("BattleScene");
GameModule.Resource.UnloadUnusedAssets();  // 整理未使用资源
GameModule.Resource.ForceUnloadUnusedAssets(performGCCollect: true);  // 强制整理+GC
```

### 多资源包

所有资源加载 API 均支持 `packageName` 可选参数，用于多资源包场景：

```csharp
var asset = await GameModule.Resource.LoadAssetAsync<TextAsset>("config", packageName: "DLC1");
var go = await GameModule.Resource.LoadGameObjectAsync("BossPrefab", parent, packageName: "DLC1");
```

---

## 三、常见错误

| 错误写法 | 正确写法 | 原因 |
|---------|---------|------|
| `SetSprite("icon", callback: sprite => {})` | `SetSprite("icon", callback: img => {})` | callback 是 `Action<Image>`，不是 `Action<Sprite>` |
| `SetSubSprite("atlas", "sub", callback: ...)` | `SetSubSprite("atlas", "sub")` | SetSubSprite 无回调重载 |
| `LoadAssetAsync<Sprite>("icon")` | `_img.SetSprite("icon")` | Sprite 应使用 SetSprite，自带缓存池 |
| `LoadAssetAsync<GameObject>` + `Instantiate` | `LoadGameObjectAsync` | 后者自动管理引用计数 |
| 忘记 `UnloadAsset` | 加载/释放必须配对 | 非 GameObject 的 Asset 需手动释放 |
| `ForceUnloadUnusedAssets(gcCollection: true)` | `ForceUnloadUnusedAssets(performGCCollect: true)` | 参数名是 `performGCCollect` |
| `SetRemoteServicesUrl("https://...")` | `SetRemoteServicesUrl("https://...", "https://fallback...")` | 需同时提供默认和备用地址 |
| `OnClose()` 释放资源 | `OnDestroy()` 释放资源 | UIWindow 无 `OnClose` 方法，销毁回调是 `OnDestroy` |
| `Resources.Load<T>(path)` | `GameModule.Resource.LoadAssetAsync<T>(location)` | 禁止 Resources.Load，必须走 YooAsset |
| `StartCoroutine(LoadRoutine())` | `await GameModule.Resource.LoadAssetAsync<T>(...)` | 禁止 Coroutine，必须用 UniTask |
| `SetSpriteAsync("icon")` | `SetSprite("icon")` | 不存在 SetSpriteAsync，SetSprite 本身内部异步 |
| `ReleaseSprite("icon")` | 无需手动释放 | SetSprite 内置缓存池，无需 ReleaseSprite |
| `LoadAssetAsync<Sprite>` + 手动释放 | `_img.SetSprite("icon")` | Sprite 加载必须用 SetSprite，禁止 LoadAssetAsync<Sprite> |

---

## 四、交叉引用

| 相关文档 | 内容 |
|---------|------|
| [ui-lifecycle.md](ui-lifecycle.md) | UIWindow 生命周期内资源释放时机 |
| [resource-patterns.md](resource-patterns.md) | 资源管理模式与生命周期进阶 |
| [event-system.md](event-system.md) | 资源加载完成的事件通知 |
| [naming-rules.md](naming-rules.md) | 禁止的代码模式 |
| [troubleshooting.md](troubleshooting.md) | 资源加载问题排查 |
