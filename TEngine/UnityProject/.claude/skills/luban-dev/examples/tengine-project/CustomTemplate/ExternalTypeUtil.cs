using UnityEngine;

/// <summary>
/// Unity 类型转换工具。将 Luban 配置类型映射为 UnityEngine 类型。
/// </summary>
public static class ExternalTypeUtil
{
    public static Vector2 NewVector2(GameConfig.vector2 v)
    {
        return new Vector2(v.X, v.Y);
    }

    public static Vector3 NewVector3(GameConfig.vector3 v)
    {
        return new Vector3(v.X, v.Y, v.Z);
    }

    public static Vector4 NewVector4(GameConfig.vector4 v)
    {
        return new Vector4(v.X, v.Y, v.Z, v.W);
    }

    public static Vector2Int NewVector2Int(GameConfig.vector2int v)
    {
        return new Vector2Int(v.X, v.Y);
    }

    public static Vector3Int NewVector3Int(GameConfig.vector3int v)
    {
        return new Vector3Int(v.X, v.Y, v.Z);
    }
}
