using System.Collections.Generic;
using System.Linq;
using GameConfig;

/// <summary>
/// 道具配置管理器。封装 TbItem 的查询逻辑，避免业务代码直接散落 ConfigSystem 调用。
/// </summary>
public class ItemConfigMgr
{
    private static ItemConfigMgr _instance;
    public static ItemConfigMgr Instance => _instance ??= new ItemConfigMgr();

    /// <summary>
    /// 获取道具配置，找不到返回 null。
    /// </summary>
    public Item GetItem(int id)
    {
        return ConfigSystem.Instance.Tables.TbItem.Get(id);
    }

    /// <summary>
    /// 获取所有道具。
    /// </summary>
    public List<Item> GetAllItems()
    {
        return ConfigSystem.Instance.Tables.TbItem.DataList;
    }

    /// <summary>
    /// 按类型筛选道具。
    /// </summary>
    public List<Item> GetItemsByType(EItemType type)
    {
        return ConfigSystem.Instance.Tables.TbItem.DataList
            .Where(i => i.Type == type)
            .ToList();
    }

    /// <summary>
    /// 按品质筛选道具。
    /// </summary>
    public List<Item> GetItemsByQuality(EQuality quality)
    {
        return ConfigSystem.Instance.Tables.TbItem.DataList
            .Where(i => i.Quality == quality)
            .ToList();
    }

    /// <summary>
    /// 获取指定品质以上的武器。
    /// </summary>
    public List<Item> GetWeaponsAboveQuality(EQuality minQuality)
    {
        return ConfigSystem.Instance.Tables.TbItem.DataList
            .Where(i => i.Type == EItemType.Weapon && i.Quality >= minQuality)
            .ToList();
    }
}
