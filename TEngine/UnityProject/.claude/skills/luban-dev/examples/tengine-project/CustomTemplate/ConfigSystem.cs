using Luban;
using GameConfig;
using TEngine;
using UnityEngine;

/// <summary>
/// 配置加载器。桥接 Luban 生成代码与 TEngine YooAsset 资源系统。
/// </summary>
public class ConfigSystem
{
    private static ConfigSystem _instance;
    public static ConfigSystem Instance => _instance ??= new ConfigSystem();

    private bool _init = false;
    private Tables _tables;

    /// <summary>
    /// 懒加载访问所有配置表。首次访问时自动加载。
    /// </summary>
    public Tables Tables
    {
        get
        {
            if (!_init)
            {
                Load();
            }
            return _tables;
        }
    }

    private IResourceModule _resourceModule;

    /// <summary>
    /// 加载所有配置表。
    /// </summary>
    public void Load()
    {
        _tables = new Tables(LoadByteBuf);
        _init = true;
    }

    /// <summary>
    /// 通过 YooAsset 加载二进制配置文件。
    /// </summary>
    private ByteBuf LoadByteBuf(string file)
    {
        if (_resourceModule == null)
        {
            _resourceModule = ModuleSystem.GetModule<IResourceModule>();
        }
        TextAsset textAsset = _resourceModule.LoadAsset<TextAsset>(file);
        return new ByteBuf(textAsset.bytes);
    }
}
