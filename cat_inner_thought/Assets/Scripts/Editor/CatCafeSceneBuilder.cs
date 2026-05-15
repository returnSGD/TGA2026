using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using System.Collections.Generic;

/// <summary>
/// Editor tool: Build a detailed Cat Cafe scene with placeholder geometry.
/// Menu: Tools > Build Cat Cafe Scene
/// All art uses Unity primitives with URP Lit materials — no imported assets needed.
/// </summary>
public class CatCafeSceneBuilder : EditorWindow
{
    // ═══════════════ Color Palette (warm, cozy café) ═══════════════
    private static readonly Color WallCream        = new(0.96f, 0.93f, 0.86f);
    private static readonly Color WallWarmBeige    = new(0.90f, 0.85f, 0.76f);
    private static readonly Color WallAccent       = new(0.82f, 0.76f, 0.68f);
    private static readonly Color TrimWhite        = new(0.94f, 0.92f, 0.88f);
    private static readonly Color WainscotWhite    = new(0.92f, 0.90f, 0.86f);
    private static readonly Color WainscotRail     = new(0.28f, 0.18f, 0.10f);
    private static readonly Color FloorWood        = new(0.48f, 0.30f, 0.16f);
    private static readonly Color FloorTile        = new(0.58f, 0.52f, 0.44f);
    private static readonly Color WoodDark         = new(0.25f, 0.15f, 0.06f);
    private static readonly Color WoodMedium       = new(0.42f, 0.26f, 0.13f);
    private static readonly Color WoodLight        = new(0.62f, 0.44f, 0.26f);
    private static readonly Color WoodCounter      = new(0.50f, 0.32f, 0.16f);
    private static readonly Color CounterMarble    = new(0.82f, 0.78f, 0.72f);
    private static readonly Color SageGreen        = new(0.45f, 0.58f, 0.42f);
    private static readonly Color OliveGreen       = new(0.38f, 0.48f, 0.32f);
    private static readonly Color LeafGreen        = new(0.35f, 0.62f, 0.35f);
    private static readonly Color Terracotta       = new(0.78f, 0.42f, 0.32f);
    private static readonly Color WarmOrange       = new(0.90f, 0.55f, 0.20f);
    private static readonly Color MustardYellow    = new(0.85f, 0.70f, 0.30f);
    private static readonly Color CushionTeal      = new(0.28f, 0.50f, 0.50f);
    private static readonly Color CushionDustyRose = new(0.80f, 0.55f, 0.55f);
    private static readonly Color CushionNavy      = new(0.22f, 0.30f, 0.45f);
    private static readonly Color CushionCream     = new(0.92f, 0.88f, 0.80f);
    private static readonly Color CushionMoss      = new(0.42f, 0.48f, 0.32f);
    private static readonly Color MetalBrass       = new(0.75f, 0.65f, 0.40f);
    private static readonly Color MetalSteel       = new(0.40f, 0.40f, 0.42f);
    private static readonly Color MetalBlack       = new(0.12f, 0.12f, 0.14f);
    private static readonly Color GlassTint        = new(0.60f, 0.78f, 0.88f, 0.30f);
    private static readonly Color Black            = new(0.06f, 0.06f, 0.06f);
    private static readonly Color OffWhite         = new(0.95f, 0.94f, 0.90f);
    private static readonly Color RugWarm          = new(0.72f, 0.60f, 0.45f);
    private static readonly Color RugPink          = new(0.82f, 0.62f, 0.58f);
    private static readonly Color CatGray          = new(0.38f, 0.38f, 0.38f);
    private static readonly Color TuxedoBlack      = new(0.08f, 0.08f, 0.10f);
    private static readonly Color PersianWhite     = new(0.94f, 0.92f, 0.88f);
    private static readonly Color TabbyOrange      = new(0.88f, 0.52f, 0.22f);
    private static readonly Color TabbyCream       = new(0.95f, 0.85f, 0.65f);
    private static readonly Color PinkInnerEar     = new(0.88f, 0.55f, 0.55f);
    private static readonly Color EyeGreen         = new(0.25f, 0.70f, 0.35f);
    private static readonly Color EyeBlue          = new(0.30f, 0.55f, 0.80f);
    private static readonly Color EyeGold          = new(0.85f, 0.70f, 0.25f);
    private static readonly Color CeilWhite        = new(0.94f, 0.92f, 0.88f);
    private static readonly Color BrickDarkRed     = new(0.55f, 0.25f, 0.18f);
    private static readonly Color BrickWarm        = new(0.60f, 0.35f, 0.25f);
    private static readonly Color StringLightWarm  = new(1f, 0.88f, 0.60f);
    private static readonly Color FireOrange       = new(1f, 0.55f, 0.15f);
    private static readonly Color Charcoal         = new(0.15f, 0.14f, 0.13f);
    private static readonly Color PillowPlum       = new(0.48f, 0.30f, 0.42f);
    private static readonly Color PillowBlush      = new(0.90f, 0.70f, 0.65f);
    private static readonly Color FrameGold        = new(0.72f, 0.58f, 0.35f);

    // ═══════════════ Room Dimensions ═══════════════
    private const float RoomW  = 14f;
    private const float RoomD  = 10f;
    private const float RoomH  = 3.5f;
    private const float WallT  = 0.2f;
    private const float FloorY = 0f;
    private const float CeilY  = RoomH;

    private static GameObject _root;

    [MenuItem("Tools/Build Cat Cafe Scene")]
    public static void BuildScene()
    {
        CleanScene();
        _root = new GameObject("CatCafe_Root");

        CreateLighting();
        CreateFloorAndRugs();
        CreateWalls();
        CreateCeilingBeams();
        CreateWainscoting();
        CreateWindows();
        CreateEntrance();
        CreateCounter();
        CreateCustomerSeating();
        CreateBoothSeating();
        CreateCozyCorner();
        CreateFireplace();
        CreateReadingNook();
        CreateCatZone();
        CreateCatWalkways();
        CreateQuietRoomDoor();
        CreateDecorations();
        CreatePlaceholderCats();
        CreateCatCafeController();
        CreateCamera();
        CreateGameManager();

        Selection.activeObject = _root;
        EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        Debug.Log("[CatCafeSceneBuilder] Scene built! Press Ctrl+S to save.");
    }

    // ═══════════════ Helpers ═══════════════

    private static void CleanScene()
    {
        var existing = GameObject.Find("CatCafe_Root");
        if (existing) Object.DestroyImmediate(existing);
        var dl = GameObject.Find("Directional Light");
        if (dl) Object.DestroyImmediate(dl);
        var mc = GameObject.Find("Main Camera");
        if (mc && mc.GetComponent<Camera>() != null) Object.DestroyImmediate(mc);
        var gm = GameObject.Find("GameManager");
        if (gm && gm.transform.parent == null) Object.DestroyImmediate(gm);
    }

    private static Shader _cachedShader;

    private static Shader GetLitShader()
    {
        if (_cachedShader != null) return _cachedShader;
        _cachedShader = Shader.Find("Universal Render Pipeline/Lit")
                     ?? Shader.Find("Standard")
                     ?? Shader.Find("Lightweight Render Pipeline/Lit");
        return _cachedShader;
    }

    private static GameObject Make(string name, PrimitiveType type, Vector3 pos, Vector3 scale, Color color, Transform parent = null)
    {
        var go = GameObject.CreatePrimitive(type);
        go.name = name;
        go.transform.position = pos;
        go.transform.localScale = scale;
        go.transform.SetParent(parent ?? _root.transform);
        var mat = new Material(GetLitShader());
        mat.color = color;
        go.GetComponent<Renderer>().material = mat;
        return go;
    }

    private static GameObject Box(string name, Vector3 pos, Vector3 size, Color color, Transform parent = null)
        => Make(name, PrimitiveType.Cube, pos, size, color, parent);

    private static GameObject Cyl(string name, Vector3 pos, float r, float h, Color color, Transform parent = null)
        => Make(name, PrimitiveType.Cylinder, pos, new Vector3(r * 2f, h / 2f, r * 2f), color, parent);

    private static GameObject Sph(string name, Vector3 pos, float r, Color color, Transform parent = null)
        => Make(name, PrimitiveType.Sphere, pos, Vector3.one * r * 2f, color, parent);

    private static GameObject Quad(string name, Vector3 pos, Vector3 size, Color color, Transform parent = null)
        => Make(name, PrimitiveType.Quad, pos, new Vector3(size.x, size.y, 1f), color, parent);

    // ═══════════════ Lighting ═══════════════

    private static void CreateLighting()
    {
        var lr = new GameObject("Lighting") { transform = { parent = _root.transform } };

        // Main directional light (warm afternoon sun through front windows)
        var sun = new GameObject("SunLight"); sun.transform.SetParent(lr.transform);
        var sl = sun.AddComponent<Light>();
        sl.type = LightType.Directional;
        sl.color = new Color(1f, 0.92f, 0.78f);
        sl.intensity = 1.2f;
        sl.shadows = LightShadows.Soft;
        sl.shadowStrength = 0.45f;
        sun.transform.rotation = Quaternion.Euler(50f, -20f, 0f);

        // Fill light (cooler ambient from back)
        var fill = new GameObject("FillLight"); fill.transform.SetParent(lr.transform);
        var fl = fill.AddComponent<Light>();
        fl.type = LightType.Directional;
        fl.color = new Color(0.75f, 0.78f, 0.85f);
        fl.intensity = 0.30f;
        fill.transform.rotation = Quaternion.Euler(30f, 150f, 0f);

        // Ceiling pendant lights — warm cozy point lights
        PendantLight("Pendant_Center",   new Vector3( 0.5f, CeilY - 0.15f,  0.2f),  lr.transform);
        PendantLight("Pendant_Left",     new Vector3(-3.5f, CeilY - 0.15f,  1.5f),  lr.transform);
        PendantLight("Pendant_Right",    new Vector3( 4.5f, CeilY - 0.15f, -2.0f),  lr.transform);
        PendantLight("Pendant_Counter",  new Vector3( 1.5f, CeilY - 0.15f, -3.5f),  lr.transform);
        PendantLight("Pendant_Window",   new Vector3(-2.0f, CeilY - 0.15f,  3.8f),  lr.transform);
        PendantLight("Pendant_Cat",      new Vector3(-5.0f, CeilY - 0.15f, -2.5f),  lr.transform);
        PendantLight("Pendant_Entrance", new Vector3( 6.0f, CeilY - 0.15f,  3.0f),  lr.transform);
    }

    private static void PendantLight(string name, Vector3 pos, Transform parent)
    {
        var go = new GameObject(name); go.transform.SetParent(parent); go.transform.position = pos;
        var light = go.AddComponent<Light>();
        light.type = LightType.Point;
        light.color = StringLightWarm;
        light.intensity = 2.2f;
        light.range = 5.5f;
        light.renderMode = LightRenderMode.ForcePixel;

        // Cord
        Cyl(name + "_cord", pos + Vector3.down * 0.25f, 0.015f, 0.5f, MetalBlack, go.transform);
        // Canopy (ceiling mount)
        Cyl(name + "_canopy", pos + Vector3.down * 0.02f, 0.06f, 0.04f, MetalBrass, go.transform);
        // Shade
        Cyl(name + "_shade", pos + Vector3.down * 0.40f, 0.14f, 0.28f, MetalBrass, go.transform);
        // Bulb glow
        Sph(name + "_bulb", pos + Vector3.down * 0.28f, 0.06f, StringLightWarm, go.transform);
    }

    // ═══════════════ Floor ═══════════════

    private static void CreateFloorAndRugs()
    {
        var fr = new GameObject("Floor") { transform = { parent = _root.transform } };

        // Main wood floor
        Box("FloorPlanks", new Vector3(0, FloorY - 0.06f, 0), new Vector3(RoomW, 0.12f, RoomD), FloorWood, fr.transform);

        // Tile area near entrance and counter
        Box("TileFloor", new Vector3(0.5f, FloorY - 0.03f, RoomD/2 - 1.2f), new Vector3(3.5f, 0.02f, 2.4f), FloorTile, fr.transform);

        // Large center rug (warm geometric)
        Box("CenterRug", new Vector3(0, FloorY + 0.01f, 0.5f), new Vector3(4.5f, 0.03f, 3.2f), RugWarm, fr.transform);

        // Cat zone rug (soft dusty rose)
        Box("CatRug", new Vector3(-4.5f, FloorY + 0.01f, -3f), new Vector3(3.5f, 0.03f, 2.8f), RugPink, fr.transform);

        // Cozy corner rug
        Box("CozyRug", new Vector3(4.5f, FloorY + 0.01f, -2.5f), new Vector3(2f, 0.03f, 1.5f), new Color(0.55f, 0.45f, 0.35f), fr.transform);

        // Reading nook rug (deep teal)
        Box("ReadingRug", new Vector3(2.2f, FloorY + 0.01f, 3.5f), new Vector3(2.5f, 0.03f, 1.8f), new Color(0.30f, 0.42f, 0.42f), fr.transform);

        // Entrance runner
        Box("EntranceRunner", new Vector3(RoomW/2 - 1.2f, FloorY + 0.01f, RoomD/2 - 1.0f), new Vector3(0.6f, 0.03f, 2.0f), new Color(0.35f, 0.28f, 0.20f), fr.transform);
    }

    // ═══════════════ Walls ═══════════════

    private static void CreateWalls()
    {
        var wr = new GameObject("Walls") { transform = { parent = _root.transform } };

        // Back wall
        Box("BackWall", new Vector3(0, RoomH/2, -RoomD/2 + WallT/2), new Vector3(RoomW, RoomH, WallT), WallCream, wr.transform);
        // Left wall
        Box("LeftWall", new Vector3(-RoomW/2 + WallT/2, RoomH/2, 0), new Vector3(WallT, RoomH, RoomD), WallCream, wr.transform);
        // Right wall
        Box("RightWall", new Vector3(RoomW/2 - WallT/2, RoomH/2, 0), new Vector3(WallT, RoomH, RoomD), WallCream, wr.transform);
        // Front wall - left pillar
        Box("FrontPillarL", new Vector3(-RoomW/2 + 0.4f, RoomH/2, RoomD/2 - WallT/2), new Vector3(0.8f, RoomH, WallT), WallCream, wr.transform);
        // Front wall - right pillar
        Box("FrontPillarR", new Vector3(RoomW/2 - 0.4f, RoomH/2, RoomD/2 - WallT/2), new Vector3(0.8f, RoomH, WallT), WallCream, wr.transform);
        // Front wall - below windows
        Box("FrontBelow", new Vector3(0, 0.45f, RoomD/2 - WallT/2), new Vector3(RoomW - 1.6f, 0.9f, WallT), WallCream, wr.transform);
        // Front wall - above windows
        Box("FrontAbove", new Vector3(0, 2.75f, RoomD/2 - WallT/2), new Vector3(RoomW - 1.6f, 1.5f, WallT), WallCream, wr.transform);
        // Ceiling
        Box("Ceiling", new Vector3(0, CeilY, 0), new Vector3(RoomW, 0.1f, RoomD), CeilWhite, wr.transform);
    }

    // ═══════════════ Ceiling Beams (exposed wood beams for rustic charm) ═══════════════

    private static void CreateCeilingBeams()
    {
        var br = new GameObject("CeilingBeams") { transform = { parent = _root.transform } };

        // Main cross beams (spanning width)
        for (int i = 0; i < 4; i++)
        {
            float z = -3.5f + i * 2.5f;
            Box($"CrossBeam_{i}", new Vector3(0, CeilY - 0.15f, z), new Vector3(RoomW - 0.4f, 0.14f, 0.18f), WoodDark, br.transform);
        }

        // Longitudinal beams
        Box("LongBeam_C", new Vector3( 0f, CeilY - 0.08f, 0),    new Vector3(0.16f, 0.16f, RoomD - 0.4f), WoodDark, br.transform);
        Box("LongBeam_L", new Vector3(-3f, CeilY - 0.08f, 0),    new Vector3(0.14f, 0.14f, RoomD - 0.4f), WoodDark, br.transform);
        Box("LongBeam_R", new Vector3( 3f, CeilY - 0.08f, 0),    new Vector3(0.14f, 0.14f, RoomD - 0.4f), WoodDark, br.transform);
    }

    // ═══════════════ Wainscoting (half-wall wood paneling) ═══════════════

    private static void CreateWainscoting()
    {
        var wr = new GameObject("Wainscoting") { transform = { parent = _root.transform } };
        float wh = 1.1f;
        float wy = wh / 2f;
        float panelW = 0.9f;
        float inset = WallT + 0.01f;

        // Back wall panels
        for (float x = -6f; x < 6.5f; x += panelW + 0.15f)
        {
            float pw = Mathf.Min(panelW, 6.5f - Mathf.Abs(x) + 0.1f);
            Box($"Panel_B_{x:F0}", new Vector3(x, wy, -RoomD/2 + inset), new Vector3(pw, wh, 0.03f), WainscotWhite, wr.transform);
        }
        // Left wall panels
        for (float z = -3.8f; z < 4.5f; z += panelW + 0.15f)
        {
            float pd = Mathf.Min(panelW, 4.5f - Mathf.Abs(z) + 0.1f);
            Box($"Panel_L_{z:F0}", new Vector3(-RoomW/2 + inset, wy, z), new Vector3(0.03f, wh, pd), WainscotWhite, wr.transform);
        }
        // Right wall panels
        for (float z = -3.8f; z < 4.5f; z += panelW + 0.15f)
        {
            float pd = Mathf.Min(panelW, 4.5f - Mathf.Abs(z) + 0.1f);
            Box($"Panel_R_{z:F0}", new Vector3(RoomW/2 - inset, wy, z), new Vector3(0.03f, wh, pd), WainscotWhite, wr.transform);
        }

        // Chair rail
        Box("RailBack",  new Vector3(0, wh, -RoomD/2 + inset + 0.02f), new Vector3(RoomW, 0.06f, 0.07f), WainscotRail, wr.transform);
        Box("RailLeft",  new Vector3(-RoomW/2 + inset + 0.02f, wh, 0), new Vector3(0.07f, 0.06f, RoomD), WainscotRail, wr.transform);
        Box("RailRight", new Vector3(RoomW/2 - inset - 0.02f, wh, 0), new Vector3(0.07f, 0.06f, RoomD), WainscotRail, wr.transform);

        // Baseboards
        Box("BaseBack",  new Vector3(0, 0.10f, -RoomD/2 + inset + 0.02f), new Vector3(RoomW, 0.20f, 0.06f), WainscotRail, wr.transform);
        Box("BaseLeft",  new Vector3(-RoomW/2 + inset + 0.02f, 0.10f, 0), new Vector3(0.06f, 0.20f, RoomD), WainscotRail, wr.transform);
        Box("BaseRight", new Vector3(RoomW/2 - inset - 0.02f, 0.10f, 0), new Vector3(0.06f, 0.20f, RoomD), WainscotRail, wr.transform);
    }

    // ═══════════════ Windows ═══════════════

    private static void CreateWindows()
    {
        var wr = new GameObject("Windows") { transform = { parent = _root.transform } };

        float paneW = 3.0f;
        float paneSpacing = 3.8f;
        float winBot = 0.9f, winTop = 2.75f, winH = winTop - winBot, winY = (winBot + winTop) / 2f;
        float z = RoomD/2 - WallT - 0.02f;
        float frameT = 0.07f;
        float mullionT = 0.04f;

        for (int i = 0; i < 3; i++)
        {
            float x = -3.8f + i * paneSpacing;
            // Glass panes (4 per window, with mullions)
            float halfPane = paneW / 2f - mullionT;
            float halfH = winH / 2f - mullionT;
            Quad($"Glass_{i}_TL", new Vector3(x - halfPane/2 - mullionT/2, winY + halfH/2 + mullionT/2, z), new Vector3(halfPane, halfH), GlassTint, wr.transform);
            Quad($"Glass_{i}_TR", new Vector3(x + halfPane/2 + mullionT/2, winY + halfH/2 + mullionT/2, z), new Vector3(halfPane, halfH), GlassTint, wr.transform);
            Quad($"Glass_{i}_BL", new Vector3(x - halfPane/2 - mullionT/2, winY - halfH/2 - mullionT/2, z), new Vector3(halfPane, halfH), GlassTint, wr.transform);
            Quad($"Glass_{i}_BR", new Vector3(x + halfPane/2 + mullionT/2, winY - halfH/2 - mullionT/2, z), new Vector3(halfPane, halfH), GlassTint, wr.transform);

            // Outer frames
            Box($"FrameT_{i}", new Vector3(x, winTop, RoomD/2 - WallT/2), new Vector3(paneW + frameT, frameT, WallT), WoodDark, wr.transform);
            Box($"FrameB_{i}", new Vector3(x, winBot, RoomD/2 - WallT/2), new Vector3(paneW + frameT, frameT, WallT), WoodDark, wr.transform);
            Box($"FrameL_{i}", new Vector3(x - paneW/2, winY, RoomD/2 - WallT/2), new Vector3(frameT, winH, WallT), WoodDark, wr.transform);
            Box($"FrameR_{i}", new Vector3(x + paneW/2, winY, RoomD/2 - WallT/2), new Vector3(frameT, winH, WallT), WoodDark, wr.transform);
            // Mullions (cross dividers)
            Box($"MullH_{i}", new Vector3(x, winY, RoomD/2 - WallT/2), new Vector3(paneW, mullionT, WallT), WoodDark, wr.transform);
            Box($"MullV_{i}", new Vector3(x, winY, RoomD/2 - WallT/2), new Vector3(mullionT, winH, WallT), WoodDark, wr.transform);
            // Sill
            Box($"Sill_{i}", new Vector3(x, winBot - 0.05f, z - 0.12f), new Vector3(paneW + 0.25f, 0.07f, 0.25f), WoodLight, wr.transform);

            // Flower box on center window
            if (i == 1)
            {
                Box($"FlowerBox_{i}", new Vector3(x, winBot + 0.02f, z - 0.15f), new Vector3(paneW - 0.3f, 0.16f, 0.17f), Terracotta, wr.transform);
                Sph($"FB_Leaf1_{i}", new Vector3(x - 0.5f, winBot + 0.22f, z - 0.15f), 0.11f, LeafGreen, wr.transform);
                Sph($"FB_Leaf2_{i}", new Vector3(x + 0.4f, winBot + 0.24f, z - 0.15f), 0.13f, SageGreen, wr.transform);
                Sph($"FB_Leaf3_{i}", new Vector3(x, winBot + 0.28f, z - 0.15f), 0.10f, OliveGreen, wr.transform);
                Sph($"FB_Leaf4_{i}", new Vector3(x - 0.15f, winBot + 0.20f, z - 0.15f), 0.08f, LeafGreen * 0.85f, wr.transform);
            }
        }

        // Side window on left wall (smaller, higher)
        Quad("SideGlass", new Vector3(-RoomW/2 + WallT + 0.02f, 1.9f, 1.5f), new Vector3(1.4f, 1.4f), GlassTint, wr.transform);
        Box("SideFrameOuter", new Vector3(-RoomW/2 + WallT/2, 1.9f, 1.5f), new Vector3(WallT, 1.5f, 1.5f), WoodDark, wr.transform);
        Box("SideSill", new Vector3(-RoomW/2 + WallT + 0.05f, 1.15f, 1.5f), new Vector3(0.07f, 0.06f, 1.2f), WoodLight, wr.transform);
        // Small plant on side window sill
        Cyl("SidePlantPot", new Vector3(-RoomW/2 + WallT + 0.1f, 1.22f, 1.2f), 0.05f, 0.10f, Terracotta, wr.transform);
        Sph("SidePlantLeaf", new Vector3(-RoomW/2 + WallT + 0.1f, 1.30f, 1.2f), 0.07f, SageGreen, wr.transform);
    }

    // ═══════════════ Entrance ═══════════════

    private static void CreateEntrance()
    {
        var er = new GameObject("Entrance") { transform = { parent = _root.transform } };

        float dx = RoomW/2 - 0.4f;
        float dz = RoomD/2 - WallT;

        // Door frame
        Box("DoorFrame", new Vector3(dx, 1.15f, dz), new Vector3(0.95f, 2.3f, WallT * 1.3f), WoodDark, er.transform);
        // Door panels (two-panel style)
        Box("DoorLeaf", new Vector3(dx, 1.15f, dz - 0.04f), new Vector3(0.80f, 2.2f, 0.07f), WoodLight, er.transform);
        Box("DoorPanel_U", new Vector3(dx, 1.55f, dz - 0.06f), new Vector3(0.60f, 0.6f, 0.02f), WoodMedium, er.transform);
        Box("DoorPanel_D", new Vector3(dx, 0.60f, dz - 0.06f), new Vector3(0.60f, 0.9f, 0.02f), WoodMedium, er.transform);
        // Handle
        Cyl("DoorHandle", new Vector3(dx + 0.22f, 1.15f, dz - 0.09f), 0.025f, 0.12f, MetalBrass, er.transform);
        // Door glass
        Quad("DoorGlass", new Vector3(dx, 1.65f, dz - 0.08f), new Vector3(0.30f, 0.50f), GlassTint, er.transform);
        // Transom window above door
        Quad("Transom", new Vector3(dx, 2.45f, dz - 0.01f), new Vector3(0.65f, 0.40f), GlassTint, er.transform);

        // Welcome mat
        Box("WelcomeMat", new Vector3(dx - 0.2f, 0.015f, dz - 0.30f), new Vector3(0.85f, 0.03f, 0.55f), new Color(0.38f, 0.26f, 0.16f), er.transform);

        // Coat rack (wall-mounted)
        var rack = new GameObject("CoatRack") { transform = { parent = er.transform, position = new Vector3(dx - 1.5f, 0, dz - 0.15f) } };
        Box("RackBoard", new Vector3(0, 1.65f, 0), new Vector3(0.6f, 0.12f, 0.03f), WoodDark, rack.transform);
        for (int i = 0; i < 4; i++)
        {
            float hx = -0.20f + i * 0.135f;
            Sph($"Hook_{i}", new Vector3(hx, 1.55f, 0.03f), 0.02f, MetalBrass, rack.transform);
        }

        // Umbrella stand
        Cyl("UmbrellaStand", new Vector3(dx - 1.2f, 0.15f, dz - 0.08f), 0.10f, 0.35f, MetalSteel, er.transform);
        for (int i = 0; i < 3; i++)
        {
            var umb = Cyl($"Umbrella_{i}", new Vector3(dx - 1.2f + i * 0.04f, 0.45f, dz - 0.08f + i * 0.03f), 0.015f, 0.65f,
                i == 0 ? CushionNavy : (i == 1 ? Terracotta : OliveGreen), er.transform);
            umb.transform.localRotation = Quaternion.Euler(3f, 0, 4f);
        }

        // Small entry console table
        Box("EntryTable", new Vector3(dx - 1.3f, 0.4f, dz - 0.25f), new Vector3(0.6f, 0.05f, 0.30f), WoodLight, er.transform);
        for (int i = 0; i < 4; i++)
        {
            float lx = (i % 2 == 0 ? -0.25f : 0.25f);
            float lz = (i < 2 ? 0.10f : -0.10f);
            Cyl($"ETLeg_{i}", new Vector3(dx - 1.3f + lx, 0.2f, dz - 0.25f + lz), 0.02f, 0.4f, WoodDark, er.transform);
        }
        // Vase + flowers
        Cyl("EntryVase", new Vector3(dx - 1.3f, 0.48f, dz - 0.25f), 0.04f, 0.14f, CushionDustyRose, er.transform);
        Sph("EntryFlower1", new Vector3(dx - 1.35f, 0.58f, dz - 0.25f), 0.05f, MustardYellow, er.transform);
        Sph("EntryFlower2", new Vector3(dx - 1.25f, 0.60f, dz - 0.25f), 0.04f, PillowBlush, er.transform);

        // Wall-mounted sign "Welcome to Cat Cafe"
        Box("WelcomeSign", new Vector3(dx - 1.3f, 2.1f, dz - 0.02f), new Vector3(0.7f, 0.35f, 0.03f), WoodLight, er.transform);
        Box("SignFrame", new Vector3(dx - 1.3f, 2.1f, dz - 0.01f), new Vector3(0.75f, 0.40f, 0.04f), WoodDark, er.transform);
    }

    // ═══════════════ Service Counter ═══════════════

    private static void CreateCounter()
    {
        var cr = new GameObject("Counter") { transform = { parent = _root.transform } };

        float cx = 1.5f, cz = -RoomD/2 + WallT + 1.0f;

        // Main counter body
        Box("CounterBody", new Vector3(cx, 0.55f, cz), new Vector3(4.0f, 1.1f, 0.65f), WoodCounter, cr.transform);
        // Counter top (marble look)
        Box("CounterTop", new Vector3(cx, 1.12f, cz - 0.02f), new Vector3(4.2f, 0.06f, 0.78f), CounterMarble, cr.transform);
        // Counter front detail panels
        for (int i = 0; i < 3; i++)
        {
            Box($"CtrPanel_{i}", new Vector3(cx - 1.2f + i * 1.2f, 0.5f, cz + 0.3f), new Vector3(0.9f, 0.7f, 0.04f), WoodDark, cr.transform);
        }

        // Display case (pastry cabinet)
        Box("DisplayCase", new Vector3(cx - 1.2f, 1.38f, cz - 0.15f), new Vector3(1.5f, 0.45f, 0.48f), MetalSteel, cr.transform);
        Box("DisplayGlass", new Vector3(cx - 1.2f, 1.38f, cz - 0.40f), new Vector3(1.3f, 0.36f, 0.04f), GlassTint, cr.transform);
        // Pastries
        Sph("Pastry1", new Vector3(cx - 1.45f, 1.28f, cz - 0.30f), 0.06f, new Color(0.75f, 0.55f, 0.35f), cr.transform);
        Sph("Pastry2", new Vector3(cx - 1.10f, 1.28f, cz - 0.30f), 0.05f, new Color(0.85f, 0.70f, 0.50f), cr.transform);
        Sph("Pastry3", new Vector3(cx - 0.95f, 1.28f, cz - 0.30f), 0.06f, new Color(0.65f, 0.40f, 0.25f), cr.transform);
        // Glass shelf inside
        Box("DCShelf", new Vector3(cx - 1.2f, 1.40f, cz - 0.30f), new Vector3(1.25f, 0.01f, 0.35f), new Color(0.7f, 0.85f, 0.9f, 0.25f), cr.transform);

        // Espresso machine
        Box("EspressoBody", new Vector3(cx + 0.5f, 1.35f, cz - 0.12f), new Vector3(0.45f, 0.40f, 0.40f), MetalSteel, cr.transform);
        Box("EspressoGroup", new Vector3(cx + 0.5f, 1.55f, cz + 0.05f), new Vector3(0.30f, 0.06f, 0.08f), MetalBlack, cr.transform);
        Cyl("DripTray", new Vector3(cx + 0.5f, 1.14f, cz + 0.10f), 0.02f, 0.02f, MetalSteel, cr.transform);
        Cyl("SteamWand", new Vector3(cx + 0.7f, 1.42f, cz + 0.05f), 0.015f, 0.16f, MetalSteel, cr.transform);

        // Coffee grinder
        Box("Grinder", new Vector3(cx + 1.15f, 1.28f, cz - 0.08f), new Vector3(0.16f, 0.22f, 0.16f), MetalSteel, cr.transform);
        Cyl("GrinderHopper", new Vector3(cx + 1.15f, 1.40f, cz - 0.08f), 0.05f, 0.06f, new Color(0.5f, 0.5f, 0.5f, 0.4f), cr.transform);

        // Hanging cups under counter
        for (int i = 0; i < 5; i++)
        {
            Cyl($"Cup_{i}", new Vector3(cx - 1.0f + i * 0.22f, 1.32f, cz - 0.35f), 0.05f, 0.10f, OffWhite, cr.transform);
            Cyl($"CupHook_{i}", new Vector3(cx - 1.0f + i * 0.22f, 1.38f, cz - 0.35f), 0.008f, 0.03f, MetalBrass, cr.transform);
        }

        // Back shelf with jars
        Box("Shelf", new Vector3(cx, 1.88f, cz - 0.35f), new Vector3(3.8f, 0.05f, 0.30f), WoodLight, cr.transform);
        for (int i = 0; i < 3; i++)
        {
            Box($"ShelfBracket_{i}", new Vector3(cx - 1.5f + i * 1.5f, 1.74f, cz - 0.47f), new Vector3(0.04f, 0.22f, 0.08f), MetalSteel, cr.transform);
            Cyl($"ShelfJar_{i}", new Vector3(cx - 1.2f + i * 1.2f, 1.98f, cz - 0.35f), 0.06f, 0.15f,
                i == 0 ? new Color(0.6f, 0.35f, 0.2f) : (i == 1 ? new Color(0.7f, 0.6f, 0.4f) : new Color(0.3f, 0.5f, 0.3f)), cr.transform);
        }
        // Small potted plant on shelf
        Cyl("ShelfPot", new Vector3(cx + 1.5f, 1.95f, cz - 0.35f), 0.045f, 0.10f, Terracotta, cr.transform);
        Sph("ShelfLeaf", new Vector3(cx + 1.5f, 2.02f, cz - 0.35f), 0.06f, LeafGreen, cr.transform);

        // Chalkboard menu on back wall
        Box("MenuFrame", new Vector3(cx + 0.5f, 2.30f, -RoomD/2 + WallT + 0.06f), new Vector3(2.15f, 1.55f, 0.05f), WoodLight, cr.transform);
        Box("MenuBoard", new Vector3(cx + 0.5f, 2.30f, -RoomD/2 + WallT + 0.08f), new Vector3(2.0f, 1.40f, 0.04f), new Color(0.12f, 0.18f, 0.12f), cr.transform);

        // Register / POS system
        Box("Register", new Vector3(cx + 1.7f, 1.22f, cz + 0.22f), new Vector3(0.22f, 0.08f, 0.18f), MetalBlack, cr.transform);
        Box("RegisterScreen", new Vector3(cx + 1.7f, 1.32f, cz + 0.12f), new Vector3(0.18f, 0.14f, 0.03f), new Color(0.2f, 0.25f, 0.35f), cr.transform);

        // Tip jar
        Cyl("TipJar", new Vector3(cx + 1.7f, 1.27f, cz + 0.36f), 0.04f, 0.10f, new Color(0.6f, 0.6f, 0.6f, 0.35f), cr.transform);
        // Coins in jar
        Cyl("Coin1", new Vector3(cx + 1.72f, 1.30f, cz + 0.36f), 0.02f, 0.008f, MetalBrass, cr.transform);
        Cyl("Coin2", new Vector3(cx + 1.68f, 1.28f, cz + 0.36f), 0.02f, 0.008f, MetalBrass, cr.transform);
    }

    // ═══════════════ Customer Seating ═══════════════

    private static void CreateCustomerSeating()
    {
        var sr = new GameObject("CustomerSeating") { transform = { parent = _root.transform } };

        // Window bar counter
        float barZ = RoomD/2 - WallT - 0.35f;
        Box("WindowBar", new Vector3(0.5f, 1.15f, barZ), new Vector3(10f, 0.06f, 0.50f), WoodLight, sr.transform);
        for (int i = 0; i < 5; i++)
        {
            Box($"BarSupp_{i}", new Vector3(-4f + i * 2f, 0.75f, barZ), new Vector3(0.06f, 0.8f, 0.08f), MetalSteel, sr.transform);
            BarStool($"BarStool_{i}", new Vector3(-4f + i * 2f, 0, barZ - 0.38f), sr.transform);
        }

        // Free-standing tables
        TableSet("Table_1", new Vector3( 3.5f, 0,  1.2f), sr.transform);
        TableSet("Table_2", new Vector3(-0.5f, 0,  2.5f), sr.transform);
        TableSet("Table_3", new Vector3(-3.5f, 0,  0.5f), sr.transform);
        TableSet("Table_4", new Vector3( 0.5f, 0, -1.5f), sr.transform);
    }

    private static void TableSet(string name, Vector3 center, Transform parent)
    {
        var tr = new GameObject(name) { transform = { parent = parent, position = center } };

        // Round wooden table with slightly tapered look
        Cyl("TableTop", new Vector3(0, 0.72f, 0), 0.42f, 0.05f, WoodLight, tr.transform);
        Cyl("TableRim", new Vector3(0, 0.70f, 0), 0.44f, 0.025f, WoodDark, tr.transform);
        Cyl("TableLeg", new Vector3(0, 0.36f, 0), 0.05f, 0.70f, WoodDark, tr.transform);
        Cyl("TableFoot", new Vector3(0, 0.03f, 0), 0.16f, 0.06f, WoodDark, tr.transform);

        // Small candle on table
        Cyl("Candle", new Vector3(0, 0.77f, 0), 0.03f, 0.08f, OffWhite, tr.transform);
        Cyl("CandleHolder", new Vector3(0, 0.75f, 0), 0.04f, 0.03f, MetalBrass, tr.transform);
        Sph("CandleFlame", new Vector3(0, 0.82f, 0), 0.02f, new Color(1f, 0.85f, 0.4f, 0.9f), tr.transform);

        // 2-3 chairs around table
        int chairCount = name == "Table_4" ? 2 : 3;
        for (int i = 0; i < chairCount; i++)
        {
            float angle = i * (360f / chairCount) * Mathf.Deg2Rad;
            float cx = Mathf.Cos(angle) * 0.58f;
            float cz = Mathf.Sin(angle) * 0.58f;
            Chair($"Chair_{i}", new Vector3(cx, 0, cz), Quaternion.Euler(0, -angle * Mathf.Rad2Deg + 90f, 0), tr.transform);
        }
    }

    private static void Chair(string name, Vector3 pos, Quaternion rot, Transform parent)
    {
        var ch = new GameObject(name) { transform = { parent = parent, localPosition = pos, localRotation = rot } };
        Box("Seat", new Vector3(0, 0.44f, 0), new Vector3(0.38f, 0.05f, 0.38f), WoodDark, ch.transform);
        Box("Back", new Vector3(0, 0.72f, -0.17f), new Vector3(0.34f, 0.50f, 0.04f), WoodDark, ch.transform);
        // Curved backrest top
        Cyl("BackTop", new Vector3(0, 0.98f, -0.17f), 0.04f, 0.34f, WoodDark, ch.transform);
        ch.transform.Find("BackTop")?.transform.SetLocalPositionAndRotation(new Vector3(0, 0.98f, -0.17f), Quaternion.Euler(0, 0, 90f));
        float l = 0.14f;
        Cyl("Leg_FL", new Vector3( l, 0.22f,  l), 0.022f, 0.44f, WoodDark, ch.transform);
        Cyl("Leg_FR", new Vector3(-l, 0.22f,  l), 0.022f, 0.44f, WoodDark, ch.transform);
        Cyl("Leg_BL", new Vector3( l, 0.22f, -l), 0.022f, 0.44f, WoodDark, ch.transform);
        Cyl("Leg_BR", new Vector3(-l, 0.22f, -l), 0.022f, 0.44f, WoodDark, ch.transform);
        Box("Cushion", new Vector3(0, 0.50f, 0), new Vector3(0.34f, 0.04f, 0.34f), CushionTeal, ch.transform);
    }

    private static void BarStool(string name, Vector3 pos, Transform parent)
    {
        var bs = new GameObject(name) { transform = { parent = parent, position = pos } };
        Cyl("Seat", new Vector3(0, 0.92f, 0), 0.17f, 0.04f, WoodDark, bs.transform);
        Cyl("SeatCushion", new Vector3(0, 0.95f, 0), 0.16f, 0.03f, CushionDustyRose, bs.transform);
        Cyl("Pole", new Vector3(0, 0.46f, 0), 0.03f, 0.90f, MetalSteel, bs.transform);
        Cyl("Base", new Vector3(0, 0.03f, 0), 0.19f, 0.06f, MetalSteel, bs.transform);
        Cyl("FootRing", new Vector3(0, 0.34f, 0), 0.15f, 0.025f, MetalBrass, bs.transform);
    }

    // ═══════════════ Booth Seating ═══════════════

    private static void CreateBoothSeating()
    {
        var br = new GameObject("BoothSeating") { transform = { parent = _root.transform } };

        float bx = -5.5f, bz = 2.5f;
        // L-shaped booth bench
        Box("BoothBackMain", new Vector3(bx, 0.5f, bz + 0.48f), new Vector3(2.4f, 0.55f, 0.10f), WoodDark, br.transform);
        Box("BoothSeatMain", new Vector3(bx, 0.22f, bz + 0.16f), new Vector3(2.4f, 0.44f, 0.58f), CushionNavy, br.transform);
        Box("BoothBackSide", new Vector3(bx - 1.25f, 0.5f, bz - 0.22f), new Vector3(0.10f, 0.55f, 1.3f), WoodDark, br.transform);
        Box("BoothSeatSide", new Vector3(bx - 0.45f, 0.22f, bz - 0.22f), new Vector3(0.58f, 0.44f, 1.1f), CushionNavy, br.transform);

        // Booth table
        Box("BoothTable", new Vector3(bx - 0.25f, 0.72f, bz - 0.05f), new Vector3(1.3f, 0.05f, 0.75f), WoodLight, br.transform);
        Box("BoothTableLeg", new Vector3(bx - 0.25f, 0.36f, bz - 0.05f), new Vector3(0.22f, 0.72f, 0.22f), WoodDark, br.transform);

        // Cushions and pillows
        Box("BoothPillow1", new Vector3(bx - 0.7f, 0.50f, bz + 0.44f), new Vector3(0.32f, 0.08f, 0.28f), CushionDustyRose, br.transform);
        Box("BoothPillow2", new Vector3(bx - 0.1f, 0.50f, bz + 0.44f), new Vector3(0.28f, 0.08f, 0.25f), MustardYellow, br.transform);
        Box("BoothPillow3", new Vector3(bx - 0.40f, 0.50f, bz - 0.30f), new Vector3(0.30f, 0.08f, 0.26f), CushionCream, br.transform);

        // Small lamp on booth table
        Cyl("BoothLampBase", new Vector3(bx - 0.45f, 0.77f, bz - 0.02f), 0.04f, 0.06f, MetalBrass, br.transform);
        Cyl("BoothLampShade", new Vector3(bx - 0.45f, 0.83f, bz - 0.02f), 0.07f, 0.14f, new Color(0.90f, 0.82f, 0.68f, 0.85f), br.transform);
    }

    // ═══════════════ Cozy Corner (sofa + bookshelf + lamp) ═══════════════

    private static void CreateCozyCorner()
    {
        var cr = new GameObject("CozyCorner") { transform = { parent = _root.transform } };
        Vector3 sp = new(4.8f, 0, -2.8f);

        // Three-seat sofa
        Box("SofaBase",   sp + new Vector3(0, 0.22f, 0.12f), new Vector3(2.6f, 0.44f, 0.75f), new Color(0.48f, 0.38f, 0.28f), cr.transform);
        Box("SofaBack",   sp + new Vector3(0, 0.55f, -0.35f), new Vector3(2.6f, 0.55f, 0.14f), new Color(0.42f, 0.32f, 0.24f), cr.transform);
        Box("SofaArmL",   sp + new Vector3(-1.25f, 0.28f, 0.12f), new Vector3(0.14f, 0.45f, 0.70f), new Color(0.42f, 0.32f, 0.24f), cr.transform);
        Box("SofaArmR",   sp + new Vector3( 1.25f, 0.28f, 0.12f), new Vector3(0.14f, 0.45f, 0.70f), new Color(0.42f, 0.32f, 0.24f), cr.transform);
        Box("Cushion_L",  sp + new Vector3(-0.65f, 0.50f, 0.10f), new Vector3(0.60f, 0.10f, 0.45f), CushionTeal, cr.transform);
        Box("Cushion_M",  sp + new Vector3( 0f, 0.50f, 0.10f), new Vector3(0.60f, 0.10f, 0.45f), CushionDustyRose, cr.transform);
        Box("Cushion_R",  sp + new Vector3( 0.65f, 0.50f, 0.10f), new Vector3(0.60f, 0.10f, 0.45f), CushionCream, cr.transform);
        // Throw pillows
        Box("Pillow_1", sp + new Vector3(-0.65f, 0.58f, 0.05f), new Vector3(0.28f, 0.07f, 0.22f), MustardYellow, cr.transform);
        Box("Pillow_2", sp + new Vector3( 0.65f, 0.58f, 0.05f), new Vector3(0.28f, 0.07f, 0.22f), PillowPlum, cr.transform);
        Box("Pillow_3", sp + new Vector3( 0f, 0.58f, 0.02f), new Vector3(0.25f, 0.07f, 0.20f), CushionMoss, cr.transform);

        // Sofa feet
        for (int i = 0; i < 4; i++)
        {
            float fx = (i % 2 == 0 ? -1.1f : 1.1f);
            float fz = (i < 2 ? -0.18f : 0.42f);
            Cyl($"SofaFoot_{i}", sp + new Vector3(fx, 0.05f, fz), 0.03f, 0.10f, WoodDark, cr.transform);
        }

        // Coffee table
        Box("CoffTableTop", new Vector3(sp.x, 0.35f, sp.z + 1.1f), new Vector3(1.2f, 0.05f, 0.7f), WoodLight, cr.transform);
        for (int i = 0; i < 4; i++)
        {
            float lx = (i % 2 == 0 ? -0.5f : 0.5f);
            float lz = (i < 2 ? 1.35f : 0.85f);
            Cyl($"CTLeg_{i}", new Vector3(sp.x + lx, 0.18f, sp.z + lz), 0.025f, 0.35f, WoodDark, cr.transform);
        }
        // Items on coffee table
        Box("CTMug", new Vector3(sp.x - 0.15f, 0.40f, sp.z + 1.15f), new Vector3(0.07f, 0.09f, 0.07f), OffWhite, cr.transform);
        Box("CTBook", new Vector3(sp.x + 0.25f, 0.39f, sp.z + 1.02f), new Vector3(0.14f, 0.018f, 0.20f), new Color(0.6f, 0.45f, 0.35f), cr.transform);

        // Bookshelf (tall, against right wall)
        var bs = new GameObject("Bookshelf") { transform = { parent = cr.transform, position = new Vector3(sp.x + 1.7f, 0, sp.z - 0.3f) } };
        Box("BS_Frame",    new Vector3(0, 1.1f, 0), new Vector3(1.05f, 2.2f, 0.32f), WoodDark, bs.transform);
        Box("BS_BackPanel", new Vector3(0, 1.1f, 0.13f), new Vector3(0.90f, 2.06f, 0.03f), WallWarmBeige, bs.transform);
        Box("BS_Top",      new Vector3(0, 2.18f, 0), new Vector3(1.02f, 0.05f, 0.30f), WoodLight, bs.transform);
        Box("BS_Bottom",   new Vector3(0, 0.05f, 0), new Vector3(1.02f, 0.10f, 0.30f), WoodDark, bs.transform);
        for (int i = 0; i < 4; i++)
        {
            Box($"Shelf_{i}", new Vector3(0, 0.42f + i * 0.52f, 0), new Vector3(0.96f, 0.04f, 0.30f), WoodLight, bs.transform);
            // Books
            for (int j = 0; j < 5; j++)
            {
                float bw = Random.Range(0.06f, 0.12f);
                Box($"Book_{i}_{j}", new Vector3(-0.32f + j * 0.16f, 0.50f + i * 0.52f, 0.04f),
                    new Vector3(bw, Random.Range(0.14f, 0.34f), 0.10f),
                    new Color(Random.Range(0.2f, 0.65f), Random.Range(0.2f, 0.48f), Random.Range(0.3f, 0.6f)), bs.transform);
            }
        }
        // Small decorative item on top
        Sph("BS_DecorTop", new Vector3(0, 2.28f, 0.05f), 0.06f, Terracotta, bs.transform);

        // Floor lamp
        var lamp = new GameObject("FloorLamp") { transform = { parent = cr.transform, position = new Vector3(sp.x - 1.7f, 0, sp.z + 0.3f) } };
        Cyl("LampPole", new Vector3(0, 0.85f, 0), 0.03f, 1.7f, MetalSteel, lamp.transform);
        Cyl("LampBase", new Vector3(0, 0.03f, 0), 0.20f, 0.06f, MetalSteel, lamp.transform);
        Cyl("LampShade", new Vector3(0, 1.78f, 0), 0.22f, 0.45f, new Color(0.92f, 0.84f, 0.70f, 0.90f), lamp.transform);
        var lampLight = new GameObject("LampLight"); lampLight.transform.SetParent(lamp.transform);
        lampLight.transform.localPosition = new Vector3(0, 1.55f, 0);
        var ll = lampLight.AddComponent<Light>();
        ll.type = LightType.Point;
        ll.color = StringLightWarm;
        ll.intensity = 1.0f;
        ll.range = 3f;
    }

    // ═══════════════ Fireplace (cozy decorative heater) ═══════════════

    private static void CreateFireplace()
    {
        var fp = new GameObject("Fireplace") { transform = { parent = _root.transform } };
        Vector3 fpos = new(-RoomW/2 + WallT + 0.5f, 0, 0.5f);

        // Hearth base
        Box("Hearth", fpos + new Vector3(0, 0.06f, 0), new Vector3(1.6f, 0.12f, 0.7f), Charcoal, fp.transform);
        // Brick surround
        Box("FPSurround", fpos + new Vector3(0, 0.75f, 0.05f), new Vector3(1.5f, 1.5f, 0.20f), BrickWarm, fp.transform);
        // Inner firebox
        Box("Firebox", fpos + new Vector3(0, 0.35f, -0.04f), new Vector3(1.0f, 0.7f, 0.12f), Charcoal, fp.transform);
        // Mantel
        Box("Mantel", fpos + new Vector3(0, 1.52f, 0.12f), new Vector3(1.7f, 0.08f, 0.25f), WoodLight, fp.transform);
        // Mantel decorations
        Cyl("MantelVaseL", fpos + new Vector3(-0.4f, 1.62f, 0.15f), 0.04f, 0.12f, CushionDustyRose, fp.transform);
        Cyl("MantelVaseR", fpos + new Vector3( 0.4f, 1.62f, 0.15f), 0.04f, 0.14f, SageGreen, fp.transform);
        var frame = new GameObject("MantelFrame") { transform = { parent = fp.transform, position = fpos + new Vector3(0, 1.65f, 0.05f) } };
        Box("MFFrame", new Vector3(0, 0, 0), new Vector3(0.40f, 0.30f, 0.02f), FrameGold, frame.transform);
        Box("MFArt", new Vector3(0, 0, -0.005f), new Vector3(0.32f, 0.22f, 0.01f), new Color(0.5f, 0.55f, 0.6f), frame.transform);

        // Fire glow light
        var fireLight = new GameObject("FireLight"); fireLight.transform.SetParent(fp.transform);
        fireLight.transform.position = fpos + new Vector3(0, 0.28f, -0.02f);
        var fLight = fireLight.AddComponent<Light>();
        fLight.type = LightType.Point;
        fLight.color = new Color(1f, 0.55f, 0.18f);
        fLight.intensity = 1.5f;
        fLight.range = 3.5f;
    }

    // ═══════════════ Reading Nook ═══════════════

    private static void CreateReadingNook()
    {
        var rn = new GameObject("ReadingNook") { transform = { parent = _root.transform } };
        float rx = 2.2f, rz = 3.8f;

        // Window-side reading bench
        Box("BenchSeat", new Vector3(rx, 0.35f, rz), new Vector3(2.0f, 0.20f, 0.55f), CushionCream, rn.transform);
        Box("BenchBase", new Vector3(rx, 0.18f, rz), new Vector3(1.9f, 0.36f, 0.50f), WoodLight, rn.transform);
        // Bench back against wall
        Box("BenchBack", new Vector3(rx, 0.80f, rz - 0.25f), new Vector3(1.9f, 0.65f, 0.08f), WoodLight, rn.transform);
        // Cushions
        Box("BenchCushion1", new Vector3(rx - 0.4f, 0.52f, rz - 0.10f), new Vector3(0.32f, 0.07f, 0.30f), CushionTeal, rn.transform);
        Box("BenchCushion2", new Vector3(rx + 0.4f, 0.52f, rz - 0.10f), new Vector3(0.32f, 0.07f, 0.30f), PillowBlush, rn.transform);

        // Small side table
        Box("SideTable", new Vector3(rx + 1.1f, 0.40f, rz + 0.05f), new Vector3(0.40f, 0.05f, 0.35f), WoodLight, rn.transform);
        Cyl("STLeg", new Vector3(rx + 1.1f, 0.20f, rz + 0.05f), 0.03f, 0.40f, WoodDark, rn.transform);
        // Lamp on side table
        Cyl("STLampBase", new Vector3(rx + 1.1f, 0.45f, rz + 0.05f), 0.04f, 0.08f, MetalBrass, rn.transform);
        Cyl("STLampShade", new Vector3(rx + 1.1f, 0.52f, rz + 0.05f), 0.08f, 0.16f, new Color(0.88f, 0.80f, 0.65f, 0.85f), rn.transform);

        // Small magazine rack
        var rack = new GameObject("MagRack") { transform = { parent = rn.transform, position = new Vector3(rx - 1.2f, 0, rz - 0.1f) } };
        Box("MRFrame", new Vector3(0, 0.28f, 0), new Vector3(0.35f, 0.56f, 0.18f), WoodDark, rack.transform);
        for (int i = 0; i < 3; i++)
        {
            Box($"Mag_{i}", new Vector3(0, 0.18f + i * 0.02f, 0.02f), new Vector3(0.25f, 0.01f, 0.14f),
                new Color(0.7f + i * 0.1f, 0.6f - i * 0.1f, 0.5f + i * 0.05f), rack.transform);
        }
    }

    // ═══════════════ Cat Furniture Zone ═══════════════

    private static void CreateCatZone()
    {
        var cz = new GameObject("CatZone") { transform = { parent = _root.transform } };

        // ── Mega Cat Tree (back-left corner) ──
        var tree = new GameObject("MegaCatTree") { transform = { parent = cz.transform, position = new Vector3(-5.2f, 0, -3.2f) } };
        Box("Base", new Vector3(0, 0.08f, 0), new Vector3(1.0f, 0.16f, 0.9f), WoodMedium, tree.transform);
        Cyl("Trunk", new Vector3(0.15f, 0.6f, 0), 0.10f, 1.2f, WoodDark, tree.transform);
        // Sisal rope wrap
        Cyl("RopeWrap", new Vector3(0.15f, 0.5f, 0), 0.13f, 0.55f, new Color(0.55f, 0.42f, 0.30f), tree.transform);
        Box("MidPlat", new Vector3(0.15f, 0.97f, 0), new Vector3(0.55f, 0.06f, 0.55f), WoodLight, tree.transform);
        // Second climbing post
        Cyl("Trunk2", new Vector3(-0.08f, 1.35f, 0.08f), 0.07f, 0.9f, WoodDark, tree.transform);
        Cyl("RopeWrap2", new Vector3(-0.08f, 1.30f, 0.08f), 0.10f, 0.40f, new Color(0.55f, 0.42f, 0.30f), tree.transform);
        Box("TopPlat", new Vector3(-0.08f, 1.68f, 0.08f), new Vector3(0.45f, 0.06f, 0.45f), WoodLight, tree.transform);
        // Top enclosed bed (cat condo)
        Box("TopBed", new Vector3(-0.08f, 1.95f, 0.08f), new Vector3(0.55f, 0.35f, 0.50f), CushionTeal, tree.transform);
        Box("TopBedInside", new Vector3(-0.08f, 1.91f, 0.08f), new Vector3(0.45f, 0.22f, 0.40f), CushionCream, tree.transform);
        // Hanging toy
        Cyl("HangString", new Vector3(0.08f, 1.40f, 0.28f), 0.008f, 0.35f, new Color(0.55f, 0.45f, 0.35f), tree.transform);
        Sph("HangBall", new Vector3(0.08f, 1.18f, 0.28f), 0.06f, WarmOrange, tree.transform);

        // ── Second Cat Tree (smaller, near window) ──
        var tree2 = new GameObject("WindowCatTree") { transform = { parent = cz.transform, position = new Vector3(-3.5f, 0, 3.2f) } };
        Box("WCTBase", new Vector3(0, 0.06f, 0), new Vector3(0.60f, 0.12f, 0.50f), WoodMedium, tree2.transform);
        Cyl("WCTPost", new Vector3(0, 0.30f, 0), 0.06f, 0.60f, WoodDark, tree2.transform);
        Cyl("WCTRope", new Vector3(0, 0.25f, 0), 0.09f, 0.30f, new Color(0.55f, 0.42f, 0.30f), tree2.transform);
        Box("WCTPlat", new Vector3(0, 0.58f, 0), new Vector3(0.45f, 0.05f, 0.40f), WoodLight, tree2.transform);
        Box("WCTBed", new Vector3(0, 0.72f, 0), new Vector3(0.45f, 0.18f, 0.38f), CushionDustyRose, tree2.transform);

        // ── Cat beds ──
        CatBed("CatBed_Blue",  new Vector3(-5.8f, 0.06f, -1.2f), CushionTeal, cz.transform);
        CatBed("CatBed_Pink",  new Vector3(-3.5f, 0.06f, -3.8f), CushionDustyRose, cz.transform);
        CatBed("CatBed_Beige", new Vector3(-4.8f, 0.97f, -2.5f), new Color(0.72f, 0.65f, 0.52f), cz.transform);
        CatBed("CatBed_Moss",  new Vector3(-5.5f, 0.06f, 1.5f), CushionMoss, cz.transform);

        // ── Scratching posts ──
        ScratchPost("Scratch_1", new Vector3(-6.2f, 0, -3.0f), cz.transform);
        ScratchPost("Scratch_2", new Vector3(-3.2f, 0, -0.8f), cz.transform);
        // Scratching board (horizontal)
        Box("ScratchBoard", new Vector3(-4.8f, 0.03f, -0.1f), new Vector3(0.6f, 0.04f, 0.25f), new Color(0.65f, 0.55f, 0.40f), cz.transform);

        // ── Cat tunnel ──
        var tunnel = new GameObject("CatTunnel") { transform = { parent = cz.transform, position = new Vector3(-4.5f, 0.08f, -0.5f) } };
        Cyl("TunnelBody", new Vector3(0, 0, 0), 0.18f, 0.40f, CushionDustyRose, tunnel.transform);
        tunnel.transform.localRotation = Quaternion.Euler(0, 0, 90f);

        // ── Food & Water Station ──
        var foodStation = new GameObject("FoodStation") { transform = { parent = cz.transform, position = new Vector3(-6.1f, 0, -0.3f) } };
        Box("Placemat", new Vector3(0, 0.025f, 0), new Vector3(0.55f, 0.02f, 0.35f), new Color(0.35f, 0.55f, 0.38f), foodStation.transform);
        CatBowl("FoodBowl", new Vector3(-0.12f, 0.05f, 0), new Color(0.70f, 0.50f, 0.30f), foodStation.transform);
        CatBowl("WaterBowl", new Vector3( 0.12f, 0.05f, 0), new Color(0.40f, 0.55f, 0.72f), foodStation.transform);

        // ── Toy area ──
        var toys = new GameObject("ToyArea") { transform = { parent = cz.transform, position = new Vector3(-4f, 0, -0.1f) } };
        Box("ToyBox", new Vector3(0, 0.15f, 0), new Vector3(0.50f, 0.30f, 0.38f), WoodLight, toys.transform);
        Box("ToyBoxLid", new Vector3(0, 0.32f, -0.05f), new Vector3(0.44f, 0.03f, 0.30f), WoodDark, toys.transform);
        // Scattered toys
        Sph("Ball_1", new Vector3( 0.28f, 0.04f,  0.40f), 0.055f, WarmOrange, toys.transform);
        Sph("Ball_2", new Vector3(-0.35f, 0.04f,  0.32f), 0.045f, SageGreen, toys.transform);
        Sph("Ball_3", new Vector3( 0.12f, 0.04f, -0.32f), 0.06f, CushionDustyRose, toys.transform);
        // Mouse toy
        var mouse = new GameObject("ToyMouse") { transform = { parent = toys.transform, position = new Vector3(-0.25f, 0.04f, -0.12f) } };
        Sph("MouseBody", new Vector3(0, 0, 0), 0.04f, CatGray, mouse.transform);
        Cyl("MouseTail", new Vector3(0, 0, -0.07f), 0.008f, 0.10f, CatGray, mouse.transform);
        Sph("MouseEarL", new Vector3(-0.02f, 0.03f, 0.02f), 0.015f, PinkInnerEar, mouse.transform);
        Sph("MouseEarR", new Vector3( 0.02f, 0.03f, 0.02f), 0.015f, PinkInnerEar, mouse.transform);
        // Feather wand
        var wand = new GameObject("FeatherWand") { transform = { parent = toys.transform, position = new Vector3(0.18f, 0.06f, -0.05f) } };
        Cyl("WandStick", new Vector3(0, 0.18f, 0), 0.012f, 0.40f, WoodLight, wand.transform);
        wand.transform.localRotation = Quaternion.Euler(0, 0, 28f);
        Sph("WandFeather", new Vector3(-0.08f, 0.38f, 0), 0.045f, CushionDustyRose, wand.transform);
        // Yarn ball
        Sph("YarnBall", new Vector3(-0.08f, 0.04f, 0.05f), 0.05f, MustardYellow, toys.transform);

        // ── Cat hammock ──
        var hammock = new GameObject("CatHammock") { transform = { parent = cz.transform, position = new Vector3(-6.3f, 1.5f, -2.5f) } };
        Box("HammockFabric", new Vector3(0, -0.04f, 0), new Vector3(0.85f, 0.03f, 0.48f), CushionTeal, hammock.transform);
        for (int i = 0; i < 4; i++)
        {
            float sx = (i % 2 == 0 ? -0.40f : 0.40f);
            float sz = (i < 2 ? -0.22f : 0.22f);
            Sph($"Hook_{i}", new Vector3(sx, 0, sz), 0.04f, MetalSteel, hammock.transform);
        }

        // ── Cat grass planter ──
        var grass = new GameObject("CatGrass") { transform = { parent = cz.transform, position = new Vector3(-6.4f, 0, 2.0f) } };
        Box("GrassPlanter", new Vector3(0, 0.08f, 0), new Vector3(0.4f, 0.16f, 0.2f), Terracotta, grass.transform);
        for (int i = 0; i < 8; i++)
        {
            float gx = -0.14f + (i % 4) * 0.09f;
            float gz = -0.06f + (i / 4) * 0.12f;
            Cyl($"Grass_{i}", new Vector3(gx, 0.16f, gz), 0.012f, Random.Range(0.06f, 0.14f), LeafGreen, grass.transform);
        }
    }

    private static void CatBed(string name, Vector3 pos, Color color, Transform parent)
    {
        var bed = new GameObject(name) { transform = { parent = parent, position = pos } };
        Cyl("Base", new Vector3(0, 0.03f, 0), 0.34f, 0.06f, color, bed.transform);
        Cyl("Rim", new Vector3(0, 0.05f, 0), 0.37f, 0.03f, color * 0.75f, bed.transform);
        Cyl("Inner", new Vector3(0, 0.04f, 0), 0.30f, 0.04f, CushionCream, bed.transform);
    }

    private static void ScratchPost(string name, Vector3 pos, Transform parent)
    {
        var sp = new GameObject(name) { transform = { parent = parent, position = pos } };
        Box("Base", new Vector3(0, 0.04f, 0), new Vector3(0.32f, 0.08f, 0.32f), WoodLight, sp.transform);
        Cyl("Post", new Vector3(0, 0.40f, 0), 0.07f, 0.80f, new Color(0.55f, 0.42f, 0.30f), sp.transform);
        Sph("TopKnob", new Vector3(0, 0.80f, 0), 0.08f, WoodLight, sp.transform);
    }

    private static void CatBowl(string name, Vector3 pos, Color innerColor, Transform parent)
    {
        var bowl = new GameObject(name) { transform = { parent = parent, position = pos } };
        Cyl("Rim", new Vector3(0, 0.02f, 0), 0.07f, 0.025f, MetalSteel, bowl.transform);
        Cyl("Inner", new Vector3(0, 0.035f, 0), 0.055f, 0.015f, innerColor, bowl.transform);
    }

    // ═══════════════ Cat Wall Walkways ═══════════════

    private static void CreateCatWalkways()
    {
        var cw = new GameObject("CatWalkways") { transform = { parent = _root.transform } };

        float shelfW = 0.30f, shelfD = 0.75f;
        float wallX = -RoomW/2 + WallT + 0.06f;

        WalkwayShelf("CatShelf_1", new Vector3(wallX, 2.0f, -1.8f), shelfW, shelfD, cw.transform);
        WalkwayShelf("CatShelf_2", new Vector3(wallX, 2.0f, -3.2f), shelfW, shelfD, cw.transform);
        WalkwayShelf("CatShelf_3", new Vector3(wallX, 2.6f, -2.5f), shelfW, 0.55f, cw.transform);

        // Bridge between shelves
        Box("Bridge", new Vector3(wallX, 2.05f, -2.5f), new Vector3(0.15f, 0.04f, 1.0f), WoodLight, cw.transform);

        // High perch on back wall
        Box("HighPerch", new Vector3(-4.5f, 2.55f, -RoomD/2 + WallT + 0.08f), new Vector3(0.65f, 0.06f, 0.38f), WoodLight, cw.transform);

        // Ramp from cat tree to shelf
        var ramp = new GameObject("Ramp") { transform = { parent = cw.transform, position = new Vector3(wallX + 0.15f, 1.4f, -2.5f) } };
        Box("RampBoard", new Vector3(0, 0.03f, 0), new Vector3(0.12f, 0.03f, 1.3f), WoodLight, ramp.transform);
        ramp.transform.localRotation = Quaternion.Euler(0, 0, -15f);
    }

    private static void WalkwayShelf(string name, Vector3 pos, float width, float depth, Transform parent)
    {
        var shelf = new GameObject(name) { transform = { parent = parent, position = pos } };
        Box("Board", Vector3.zero, new Vector3(width, 0.04f, depth), WoodLight, shelf.transform);
        Box("BracketF", new Vector3(0, -0.1f,  depth/2 - 0.1f), new Vector3(0.04f, 0.16f, 0.05f), MetalSteel, shelf.transform);
        Box("BracketB", new Vector3(0, -0.1f, -depth/2 + 0.1f), new Vector3(0.04f, 0.16f, 0.05f), MetalSteel, shelf.transform);
    }

    // ═══════════════ Quiet Room Door (back wall) ═══════════════

    private static void CreateQuietRoomDoor()
    {
        var qr = new GameObject("QuietRoomEntrance") { transform = { parent = _root.transform } };

        float dx = -5.8f;
        float dz = -RoomD/2 + WallT;

        // Door frame
        Box("QRFrame", new Vector3(dx, 1.15f, dz), new Vector3(0.90f, 2.3f, WallT * 1.4f), WoodDark, qr.transform);
        // Door (slightly ajar, hinting at rooms beyond)
        Box("QRDoor", new Vector3(dx, 1.15f, dz - 0.04f), new Vector3(0.76f, 2.2f, 0.07f), WoodLight, qr.transform);
        // Door handle
        Cyl("QRHandle", new Vector3(dx + 0.20f, 1.15f, dz - 0.09f), 0.025f, 0.10f, MetalBrass, qr.transform);
        // Sign: "静音隔间 / Quiet Room"
        Box("QRSign", new Vector3(dx, 1.95f, dz - 0.06f), new Vector3(0.55f, 0.20f, 0.03f), WoodLight, qr.transform);
        // Light leaking from under door
        var underLight = new GameObject("QRUnderLight"); underLight.transform.SetParent(qr.transform);
        underLight.transform.position = new Vector3(dx, 0.05f, dz - 0.35f);
        var ul = underLight.AddComponent<Light>();
        ul.type = LightType.Point;
        ul.color = new Color(0.95f, 0.80f, 0.55f);
        ul.intensity = 0.4f;
        ul.range = 1.5f;
    }

    // ═══════════════ Decorations ═══════════════

    private static void CreateDecorations()
    {
        var dr = new GameObject("Decorations") { transform = { parent = _root.transform } };

        // ── Plants ──
        PottedPlant("Plant_Corner",   new Vector3(-6.3f, 0,  4.2f), 0.55f, dr.transform);
        PottedPlant("Plant_Entrance", new Vector3( 5.8f, 0,  3.8f), 0.40f, dr.transform);
        PottedPlant("Plant_Counter",  new Vector3( 0.0f, 0, -4.0f), 0.35f, dr.transform);
        PottedPlant("Plant_Cozy",     new Vector3( 3.0f, 0, -3.8f), 0.45f, dr.transform);
        PottedPlant("Plant_BackWall", new Vector3(-2.0f, 0, -4.2f), 0.5f, dr.transform);

        // Hanging plant from ceiling
        var hangingPlant = new GameObject("HangingPlant") { transform = { parent = dr.transform, position = new Vector3(-1.5f, CeilY - 0.1f, -RoomD/2 + WallT + 0.1f) } };
        Cyl("HPRope", new Vector3(0, -0.40f, 0), 0.008f, 0.80f, new Color(0.55f, 0.45f, 0.35f), hangingPlant.transform);
        Cyl("HPPot", new Vector3(0, -0.80f, 0), 0.12f, 0.22f, Terracotta, hangingPlant.transform);
        for (int i = 0; i < 6; i++)
        {
            float angle = i * 60f * Mathf.Deg2Rad;
            Sph($"HPLeaf_{i}", new Vector3(Mathf.Cos(angle) * 0.16f, -0.62f + i * 0.05f, Mathf.Sin(angle) * 0.10f), 0.10f, LeafGreen, hangingPlant.transform);
        }

        // ── Wall Art ──
        WallFrame("Frame_Landscape", new Vector3(-2.5f, 2.2f, -RoomD/2 + WallT + 0.07f), 0.9f, 0.7f, new Color(0.4f, 0.55f, 0.5f), dr.transform);
        WallFrame("Frame_Cat1",      new Vector3( 2.5f, 2.35f, -RoomD/2 + WallT + 0.07f), 0.5f, 0.6f, new Color(0.6f, 0.45f, 0.35f), dr.transform);
        WallFrame("Frame_Cat2",      new Vector3( 3.4f, 2.2f, -RoomD/2 + WallT + 0.07f), 0.4f, 0.5f, new Color(0.5f, 0.5f, 0.55f), dr.transform);
        WallFrame("Frame_Abstract",  new Vector3(-4.5f, 2.1f, -RoomD/2 + WallT + 0.07f), 0.45f, 0.55f, new Color(0.7f, 0.55f, 0.4f), dr.transform);
        WallFrame("Frame_Warm",      new Vector3( 0.5f, 2.4f, -RoomD/2 + WallT + 0.07f), 0.55f, 0.45f, new Color(0.65f, 0.50f, 0.38f), dr.transform);

        // Side wall art
        WallFrame("Frame_Side1", new Vector3(-RoomW/2 + WallT + 0.07f, 2.4f, -1.5f), 0.5f, 0.65f, new Color(0.35f, 0.45f, 0.55f), dr.transform);
        WallFrame("Frame_Side2", new Vector3(-RoomW/2 + WallT + 0.07f, 2.3f,  3.0f), 0.4f, 0.55f, new Color(0.55f, 0.40f, 0.45f), dr.transform);

        // ── Wall Clock ──
        var clock = new GameObject("WallClock") { transform = { parent = dr.transform, position = new Vector3(3.8f, 2.7f, -RoomD/2 + WallT + 0.06f) } };
        Cyl("ClockBody", new Vector3(0, 0, 0), 0.22f, 0.03f, OffWhite, clock.transform);
        Cyl("ClockCenter", new Vector3(0, 0, 0.02f), 0.03f, 0.03f, MetalBlack, clock.transform);
        for (int i = 0; i < 12; i++)
        {
            float angle = i * 30f * Mathf.Deg2Rad;
            Sph($"Marker_{i}", new Vector3(Mathf.Cos(angle) * 0.17f, Mathf.Sin(angle) * 0.17f, 0.02f), 0.014f, MetalBlack, clock.transform);
        }
        // Cat ears on clock
        var earL = Cyl("ClockEarL", new Vector3(-0.10f, 0.18f, 0), 0.03f, 0.06f, WoodDark, clock.transform);
        earL.transform.localRotation = Quaternion.Euler(0, 0, 15f);
        var earR = Cyl("ClockEarR", new Vector3( 0.10f, 0.18f, 0), 0.03f, 0.06f, WoodDark, clock.transform);
        earR.transform.localRotation = Quaternion.Euler(0, 0, -15f);

        // ── String lights ──
        var stringLights = new GameObject("StringLights") { transform = { parent = dr.transform } };
        for (int i = 0; i < 10; i++)
        {
            float x = -5.5f + i * 1.2f;
            float z = 1.8f + Mathf.Sin(i * 0.7f) * 1.4f;
            Sph($"SL_Bulb_{i}", new Vector3(x, CeilY - 0.28f, z), 0.04f, StringLightWarm, stringLights.transform);
            Cyl($"SL_Cord_{i}", new Vector3(x, CeilY - 0.12f, z), 0.005f, 0.28f, MetalBlack, stringLights.transform);
        }

        // ── Notice board ──
        var noticeBoard = new GameObject("NoticeBoard") { transform = { parent = dr.transform, position = new Vector3(RoomW/2 - 0.6f, 1.8f, RoomD/2 - WallT - 0.8f) } };
        Box("NBBorder", new Vector3(0, 0, 0), new Vector3(0.62f, 0.85f, 0.04f), WoodLight, noticeBoard.transform);
        Box("NBCork", new Vector3(0, 0, -0.01f), new Vector3(0.54f, 0.76f, 0.02f), new Color(0.65f, 0.50f, 0.35f), noticeBoard.transform);
        // Notes
        Box("NBNote1", new Vector3(-0.12f, 0.12f, -0.03f), new Vector3(0.09f, 0.11f, 0.005f), new Color(1f, 0.88f, 0.55f), noticeBoard.transform);
        Box("NBNote2", new Vector3( 0.18f, -0.08f, -0.03f), new Vector3(0.11f, 0.09f, 0.005f), new Color(0.65f, 0.85f, 1f), noticeBoard.transform);
        Box("NBNote3", new Vector3(-0.05f, -0.15f, -0.03f), new Vector3(0.08f, 0.08f, 0.005f), new Color(1f, 0.70f, 0.75f), noticeBoard.transform);
        // Push pins
        Sph("NBPin1", new Vector3(-0.12f, 0.20f, -0.04f), 0.012f, MetalBrass, noticeBoard.transform);
        Sph("NBPin2", new Vector3( 0.18f, 0.00f, -0.04f), 0.012f, MetalBrass, noticeBoard.transform);
        Sph("NBPin3", new Vector3(-0.05f, -0.08f, -0.04f), 0.012f, MetalBrass, noticeBoard.transform);

        // ── Speaker ──
        Box("Speaker", new Vector3(-RoomW/2 + WallT + 0.25f, 2.5f, -3.5f), new Vector3(0.16f, 0.24f, 0.13f), MetalBlack, dr.transform);

        // ── Curtains on side window ──
        var curtainL = new GameObject("CurtainL") { transform = { parent = dr.transform, position = new Vector3(-RoomW/2 + WallT + 0.06f, 1.9f, 0.9f) } };
        Box("CFabricL", new Vector3(0, 0, 0), new Vector3(0.04f, 1.7f, 0.55f), new Color(0.55f, 0.45f, 0.38f, 0.80f), curtainL.transform);
        var curtainR = new GameObject("CurtainR") { transform = { parent = dr.transform, position = new Vector3(-RoomW/2 + WallT + 0.06f, 1.9f, 2.1f) } };
        Box("CFabricR", new Vector3(0, 0, 0), new Vector3(0.04f, 1.7f, 0.55f), new Color(0.55f, 0.45f, 0.38f, 0.80f), curtainR.transform);
        // Curtain rod
        Cyl("CurtainRod", new Vector3(-RoomW/2 + WallT + 0.06f, 2.75f, 1.5f), 0.02f, 1.8f, WoodDark, dr.transform);
        dr.transform.Find("CurtainRod")?.transform.SetLocalPositionAndRotation(
            new Vector3(-RoomW/2 + WallT + 0.06f, 2.75f, 1.5f), Quaternion.Euler(0, 0, 90f));

        // ── Cat-themed wall decals (simple colored squares) ──
        Box("Decal_Paw1", new Vector3(-RoomW/2 + WallT + 0.04f, 1.35f, -3.0f), new Vector3(0.02f, 0.18f, 0.18f), CushionDustyRose, dr.transform);
        Box("Decal_Paw2", new Vector3(-RoomW/2 + WallT + 0.04f, 1.35f,  4.0f), new Vector3(0.02f, 0.15f, 0.15f), CushionTeal, dr.transform);
    }

    private static void PottedPlant(string name, Vector3 pos, float scale, Transform parent)
    {
        var plant = new GameObject(name) { transform = { parent = parent, position = pos } };
        float s = scale;
        // Pot
        Cyl("Pot", new Vector3(0, 0.12f * s, 0), 0.16f * s, 0.28f * s, Terracotta, plant.transform);
        // Pot rim
        Cyl("PotRim", new Vector3(0, 0.26f * s, 0), 0.18f * s, 0.03f * s, Terracotta * 0.85f, plant.transform);
        // Soil
        Cyl("Soil", new Vector3(0, 0.27f * s, 0), 0.14f * s, 0.02f * s, new Color(0.25f, 0.15f, 0.08f), plant.transform);
        // Leaves
        Sph("Leaf_Big",    new Vector3(0, 0.38f * s, 0), 0.20f * s, LeafGreen, plant.transform);
        Sph("Leaf_Med1",   new Vector3(0.12f * s, 0.43f * s, 0.07f * s), 0.14f * s, SageGreen, plant.transform);
        Sph("Leaf_Med2",   new Vector3(-0.10f * s, 0.41f * s, -0.08f * s), 0.13f * s, OliveGreen, plant.transform);
        Sph("Leaf_Sm1",    new Vector3(0.06f * s, 0.50f * s, -0.04f * s), 0.10f * s, LeafGreen * 0.85f, plant.transform);
        Sph("Leaf_Sm2",    new Vector3(-0.06f * s, 0.48f * s, 0.05f * s), 0.09f * s, SageGreen * 0.9f, plant.transform);
        Sph("Leaf_Sm3",    new Vector3(0.03f * s, 0.54f * s, 0.01f * s), 0.08f * s, OliveGreen * 0.9f, plant.transform);
    }

    private static void WallFrame(string name, Vector3 pos, float w, float h, Color artColor, Transform parent)
    {
        var frame = new GameObject(name) { transform = { parent = parent, position = pos } };
        Box("OuterFrame", Vector3.zero, new Vector3(w, h, 0.04f), FrameGold, frame.transform);
        Box("InnerFrame", new Vector3(0, 0, -0.005f), new Vector3(w - 0.06f, h - 0.06f, 0.02f), WoodLight, frame.transform);
        Box("Mat", new Vector3(0, 0, -0.01f), new Vector3(w - 0.12f, h - 0.12f, 0.01f), OffWhite, frame.transform);
        Box("Art", new Vector3(0, 0, -0.015f), new Vector3(w - 0.20f, h - 0.20f, 0.005f), artColor, frame.transform);
    }

    // ═══════════════ Placeholder Cats ═══════════════

    private static void CreatePlaceholderCats()
    {
        var catsRoot = new GameObject("Cats") { transform = { parent = _root.transform } };

        // ── Oreo (Tuxedo) - Tsundere ──
        var oreo = new GameObject("Cat_Oreo") { transform = { parent = catsRoot.transform, position = new Vector3(-3.5f, 0.45f, -2.2f) } };
        BuildCat(oreo.transform, TuxedoBlack, OffWhite, EyeGreen, "黑白奶牛猫");
        oreo.AddComponent<CatPlaceholder>().Init("奥利奥", "Oreo", "傲娇的奶牛猫，嘴上说不要身体很诚实");

        // ── Xiaoxue (Persian) - Timid ──
        var xue = new GameObject("Cat_Xiaoxue") { transform = { parent = catsRoot.transform, position = new Vector3(4.0f, 0.45f, -3.2f) } };
        BuildCat(xue.transform, PersianWhite, new Color(0.88f, 0.85f, 0.80f), EyeBlue, "白色波斯猫");
        xue.AddComponent<CatPlaceholder>().Init("小雪", "Xiaoxue", "胆小的波斯猫，世界好可怕但你好温暖");

        // ── Orange (Tabby) - Foodie ──
        var orange = new GameObject("Cat_Orange") { transform = { parent = catsRoot.transform, position = new Vector3(-5.5f, 0.06f, -0.8f) } };
        BuildCat(orange.transform, TabbyOrange, TabbyCream, EyeGold, "橘猫");
        orange.AddComponent<CatPlaceholder>().Init("橘子", "Orange", "贪吃鬼橘猫，吃饱睡睡饱吃，给吃的就是好朋友");
    }

    private static void BuildCat(Transform parent, Color bodyColor, Color bellyColor, Color eyeColor, string desc)
    {
        // Body
        var body = Sph("Body", Vector3.zero, 1f, bodyColor, parent);
        body.transform.localScale = new Vector3(0.26f, 0.21f, 0.36f);

        // Head
        var head = new GameObject("Head") { transform = { parent = parent, localPosition = new Vector3(0, 0.23f, 0.16f) } };
        Sph("Skull", new Vector3(0, 0, 0), 0.18f, bodyColor, head.transform);
        Sph("CheekL", new Vector3(-0.10f, -0.04f, 0.08f), 0.07f, bodyColor, head.transform);
        Sph("CheekR", new Vector3( 0.10f, -0.04f, 0.08f), 0.07f, bodyColor, head.transform);
        // Snout
        Sph("Snout", new Vector3(0, -0.05f, 0.12f), 0.06f, bellyColor, head.transform);
        // Nose
        Sph("Nose", new Vector3(0, -0.035f, 0.17f), 0.022f, PinkInnerEar, head.transform);
        // Eyes
        Sph("EyeL", new Vector3(-0.06f, 0.01f, 0.15f), 0.04f, eyeColor, head.transform);
        Sph("EyeR", new Vector3( 0.06f, 0.01f, 0.15f), 0.04f, eyeColor, head.transform);
        // Pupils
        Sph("PupilL", new Vector3(-0.06f, 0.01f, 0.18f), 0.018f, Black, head.transform);
        Sph("PupilR", new Vector3( 0.06f, 0.01f, 0.18f), 0.018f, Black, head.transform);
        // Ears
        var earL = Cyl("EarL", new Vector3(-0.10f, 0.10f, 0.04f), 0.04f, 0.12f, bodyColor, head.transform);
        earL.transform.localRotation = Quaternion.Euler(12f, 0, 18f);
        var earR = Cyl("EarR", new Vector3( 0.10f, 0.10f, 0.04f), 0.04f, 0.12f, bodyColor, head.transform);
        earR.transform.localRotation = Quaternion.Euler(12f, 0, -18f);
        // Inner ears
        var inL = Cyl("EarInnerL", new Vector3(-0.10f, 0.10f, 0.05f), 0.022f, 0.07f, PinkInnerEar, head.transform);
        inL.transform.localRotation = Quaternion.Euler(12f, 0, 18f);
        var inR = Cyl("EarInnerR", new Vector3( 0.10f, 0.10f, 0.05f), 0.022f, 0.07f, PinkInnerEar, head.transform);
        inR.transform.localRotation = Quaternion.Euler(12f, 0, -18f);

        // Belly
        var belly = Sph("Belly", new Vector3(0, -0.06f, -0.02f), 1f, bellyColor, parent);
        belly.transform.localScale = new Vector3(0.21f, 0.15f, 0.26f);

        // Tail
        var tail = Cyl("Tail", new Vector3(0, 0.04f, -0.22f), 0.03f, 0.28f, bodyColor, parent);
        tail.transform.localRotation = Quaternion.Euler(35f, 0, 22f);

        // Paws
        Sph("Paw_FL", new Vector3(-0.10f, -0.13f,  0.13f), 0.048f, bellyColor, parent);
        Sph("Paw_FR", new Vector3( 0.10f, -0.13f,  0.13f), 0.048f, bellyColor, parent);
        Sph("Paw_BL", new Vector3(-0.10f, -0.13f, -0.13f), 0.048f, bellyColor, parent);
        Sph("Paw_BR", new Vector3( 0.10f, -0.13f, -0.13f), 0.048f, bellyColor, parent);

        // Whiskers
        for (int i = 0; i < 3; i++)
        {
            float a = (-1 + i) * 25f;
            var wL = Cyl($"Whisker_L{i}", new Vector3(-0.08f, -0.02f, 0.13f), 0.004f, 0.10f, new Color(0.8f, 0.8f, 0.8f, 0.7f), head.transform);
            wL.transform.localRotation = Quaternion.Euler(0, a, 10f);
            var wR = Cyl($"Whisker_R{i}", new Vector3( 0.08f, -0.02f, 0.13f), 0.004f, 0.10f, new Color(0.8f, 0.8f, 0.8f, 0.7f), head.transform);
            wR.transform.localRotation = Quaternion.Euler(0, -a, -10f);
        }
    }

    // ═══════════════ Camera ═══════════════

    private static void CreateCamera()
    {
        var cam = new GameObject("Main Camera");
        cam.transform.position = new Vector3(0, 6.8f, -3.0f);
        cam.transform.rotation = Quaternion.Euler(58f, 0f, 0f);
        var c = cam.AddComponent<Camera>();
        c.fieldOfView = 52f;
        c.backgroundColor = new Color(0.16f, 0.18f, 0.20f);
        cam.AddComponent<AudioListener>();
        cam.AddComponent<SimpleCameraController>();
        cam.tag = "MainCamera";
    }

    private static void CreateCatCafeController()
    {
        _root.AddComponent<CatCafeController>();
    }

    private static void CreateGameManager()
    {
        var gm = new GameObject("GameManager");
        gm.transform.SetParent(_root.transform);
        gm.AddComponent<GameManager>();
    }
}
