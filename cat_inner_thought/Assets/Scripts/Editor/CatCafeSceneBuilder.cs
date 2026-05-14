using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine.Rendering;
using System.Collections.Generic;

/// <summary>
/// Editor tool: Build the Cat Cafe scene with placeholder geometry.
/// Menu: Tools > Build Cat Cafe Scene
/// </summary>
public class CatCafeSceneBuilder : EditorWindow
{
    // ── Color palette ──
    private static readonly Color WarmWood      = new(0.55f, 0.35f, 0.18f);
    private static readonly Color LightWood     = new(0.72f, 0.52f, 0.32f);
    private static readonly Color CreamWall     = new(0.96f, 0.92f, 0.82f);
    private static readonly Color DarkBrown     = new(0.25f, 0.15f, 0.05f);
    private static readonly Color SoftGreen     = new(0.45f, 0.65f, 0.40f);
    private static readonly Color CatGray       = new(0.35f, 0.35f, 0.35f);
    private static readonly Color WarmOrange    = new(0.90f, 0.55f, 0.20f);
    private static readonly Color White         = new(0.95f, 0.95f, 0.93f);
    private static readonly Color Black         = new(0.08f, 0.08f, 0.08f);
    private static readonly Color CushionBlue   = new(0.40f, 0.55f, 0.70f);
    private static readonly Color CushionPink   = new(0.88f, 0.65f, 0.65f);
    private static readonly Color MetalGray     = new(0.45f, 0.45f, 0.48f);
    private static readonly Color CounterTop    = new(0.60f, 0.55f, 0.50f);
    private static readonly Color BrickRed      = new(0.60f, 0.28f, 0.20f);
    private static readonly Color GlassBlue     = new(0.55f, 0.75f, 0.85f, 0.35f);
    private static readonly Color RugBeige      = new(0.78f, 0.70f, 0.58f);
    private static readonly Color CeilingWhite  = new(0.92f, 0.90f, 0.86f);

    // ── Layout constants (meters) ──
    private const float RoomW  = 14f;
    private const float RoomD  = 10f;
    private const float RoomH  = 3.5f;
    private const float WallT  = 0.2f;
    private const float FloorY = 0f;
    private const float CeilY  = RoomH;

    private static GameObject _root;
    private static Material _defaultMat;

    [MenuItem("Tools/Build Cat Cafe Scene")]
    public static void BuildScene()
    {
        CleanScene();
        _root = new GameObject("CatCafe_Root");

        CreateLighting();
        CreateFloor();
        CreateWalls();
        CreateWindow();
        CreateCounter();
        CreateCustomerSeating();
        CreateCatFurnitureZone();
        CreateCozyCorner();
        CreateCafeDecor();
        CreatePlaceholderCats();
        CreateEntrance();
        CreateCamera();
        CreateGameManager();

        Selection.activeObject = _root;
        EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        Debug.Log("[CatCafeSceneBuilder] Scene built successfully! Press Ctrl+S to save.");
    }

    // ────────────────────────────────

    private static void CleanScene()
    {
        // Remove existing root if present
        var existing = GameObject.Find("CatCafe_Root");
        if (existing) DestroyImmediate(existing);
        // Remove default directional light
        var dl = GameObject.Find("Directional Light");
        if (dl) DestroyImmediate(dl);
        // Remove default Main Camera (we create our own)
        var mc = GameObject.Find("Main Camera");
        if (mc && mc.GetComponent<Camera>() != null) DestroyImmediate(mc);
    }

    // ────────────────────────────────

    private static GameObject CreateBox(string name, Vector3 pos, Vector3 size, Color color, Transform parent = null)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = name;
        go.transform.position = pos;
        go.transform.localScale = size;
        go.transform.SetParent(parent ?? _root.transform);
        var mat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        mat.color = color;
        go.GetComponent<Renderer>().material = mat;
        return go;
    }

    private static GameObject CreateCylinder(string name, Vector3 pos, float radius, float height, Color color, Transform parent = null)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        go.name = name;
        go.transform.position = pos;
        go.transform.localScale = new Vector3(radius * 2f, height / 2f, radius * 2f);
        go.transform.SetParent(parent ?? _root.transform);
        var mat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        mat.color = color;
        go.GetComponent<Renderer>().material = mat;
        return go;
    }

    private static GameObject CreateSphere(string name, Vector3 pos, float radius, Color color, Transform parent = null)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        go.name = name;
        go.transform.position = pos;
        go.transform.localScale = Vector3.one * radius * 2f;
        go.transform.SetParent(parent ?? _root.transform);
        var mat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        mat.color = color;
        go.GetComponent<Renderer>().material = mat;
        return go;
    }

    private static GameObject CreateQuad(string name, Vector3 pos, Vector3 size, Color color, Transform parent = null)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Quad);
        go.name = name;
        go.transform.position = pos;
        go.transform.localScale = new Vector3(size.x, size.y, 1f);
        go.transform.SetParent(parent ?? _root.transform);
        var mat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        mat.color = color;
        go.GetComponent<Renderer>().material = mat;
        return go;
    }

    // ────────────────────────────────

    private static void CreateLighting()
    {
        var lightRoot = new GameObject("Lighting") { transform = { parent = _root.transform } };

        // Warm ambient directional light (simulating window light)
        var sun = new GameObject("SunLight");
        sun.transform.SetParent(lightRoot.transform);
        var sl = sun.AddComponent<Light>();
        sl.type = LightType.Directional;
        sl.color = new Color(1f, 0.95f, 0.85f);
        sl.intensity = 1.2f;
        sl.shadows = LightShadows.Soft;
        sun.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

        // Fill light from opposite side
        var fill = new GameObject("FillLight");
        fill.transform.SetParent(lightRoot.transform);
        var fl = fill.AddComponent<Light>();
        fl.type = LightType.Directional;
        fl.color = new Color(0.8f, 0.82f, 0.9f);
        fl.intensity = 0.4f;
        fill.transform.rotation = Quaternion.Euler(30f, 150f, 0f);

        // Warm ceiling point lights (cozy café feel)
        CreateCeilingLight("CeilingLight1", new Vector3(  0f, CeilY - 0.1f,  2f), lightRoot.transform);
        CreateCeilingLight("CeilingLight2", new Vector3(  0f, CeilY - 0.1f, -2f), lightRoot.transform);
        CreateCeilingLight("CeilingLight3", new Vector3( -4f, CeilY - 0.1f,  1f), lightRoot.transform);
        CreateCeilingLight("CeilingLight4", new Vector3(  4f, CeilY - 0.1f,  1f), lightRoot.transform);
    }

    private static void CreateCeilingLight(string name, Vector3 pos, Transform parent)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent);
        go.transform.position = pos;
        var light = go.AddComponent<Light>();
        light.type = LightType.Point;
        light.color = new Color(1f, 0.85f, 0.65f);
        light.intensity = 2.5f;
        light.range = 6f;
        light.renderMode = LightRenderMode.ForcePixel;

        // Small pendant lamp mesh
        var fixture = CreateCylinder(name + "_fixture", pos + Vector3.down * 0.3f, 0.12f, 0.15f, MetalGray, go.transform);
        var bulb = CreateSphere(name + "_bulb", pos + Vector3.down * 0.15f, 0.1f, new Color(1f, 0.95f, 0.7f), go.transform);
    }

    // ────────────────────────────────

    private static void CreateFloor()
    {
        var floorRoot = new GameObject("Floor") { transform = { parent = _root.transform } };

        // Main floor
        var floor = CreateBox("FloorPlane", new Vector3(0, FloorY - 0.05f, 0),
            new Vector3(RoomW, 0.1f, RoomD), WarmWood, floorRoot.transform);

        // Large area rug in center
        CreateBox("CenterRug", new Vector3(0, FloorY, 0),
            new Vector3(5f, 0.04f, 3.5f), RugBeige, floorRoot.transform);

        // Small rug in cat zone
        CreateBox("CatZoneRug", new Vector3(-4.5f, FloorY, -3f),
            new Vector3(3f, 0.04f, 2.5f), CushionPink * 0.7f, floorRoot.transform);
    }

    // ────────────────────────────────

    private static void CreateWalls()
    {
        var wallRoot = new GameObject("Walls") { transform = { parent = _root.transform } };

        // Back wall (-Z)
        CreateBox("BackWall", new Vector3(0, RoomH/2, -RoomD/2 + WallT/2),
            new Vector3(RoomW, RoomH, WallT), CreamWall, wallRoot.transform);

        // Left wall (-X)
        CreateBox("LeftWall", new Vector3(-RoomW/2 + WallT/2, RoomH/2, 0),
            new Vector3(WallT, RoomH, RoomD), CreamWall, wallRoot.transform);

        // Right wall (+X)
        CreateBox("RightWall", new Vector3(RoomW/2 - WallT/2, RoomH/2, 0),
            new Vector3(WallT, RoomH, RoomD), CreamWall, wallRoot.transform);

        // Front wall bottom section (below window)
        CreateBox("FrontWallBottom", new Vector3(0, 0.5f, RoomD/2 - WallT/2),
            new Vector3(RoomW, 1f, WallT), CreamWall, wallRoot.transform);

        // Front wall top section (above window)
        CreateBox("FrontWallTop", new Vector3(0, 2.75f, RoomD/2 - WallT/2),
            new Vector3(RoomW, 1.5f, WallT), CreamWall, wallRoot.transform);

        // Front wall left column
        CreateBox("FrontWallLeftCol", new Vector3(-RoomW/2 + 0.5f, RoomH/2, RoomD/2 - WallT/2),
            new Vector3(1f, RoomH, WallT), CreamWall, wallRoot.transform);

        // Front wall right column
        CreateBox("FrontWallRightCol", new Vector3(RoomW/2 - 0.5f, RoomH/2, RoomD/2 - WallT/2),
            new Vector3(1f, RoomH, WallT), CreamWall, wallRoot.transform);

        // Ceiling
        CreateBox("Ceiling", new Vector3(0, CeilY, 0),
            new Vector3(RoomW, 0.1f, RoomD), CeilingWhite, wallRoot.transform);

        // Baseboard back
        CreateBox("BaseboardBack", new Vector3(0, 0.12f, -RoomD/2 + WallT + 0.01f),
            new Vector3(RoomW, 0.24f, 0.05f), DarkBrown, wallRoot.transform);
        // Baseboard left
        CreateBox("BaseboardLeft", new Vector3(-RoomW/2 + WallT + 0.01f, 0.12f, 0),
            new Vector3(0.05f, 0.24f, RoomD), DarkBrown, wallRoot.transform);
        // Baseboard right
        CreateBox("BaseboardRight", new Vector3(RoomW/2 - WallT - 0.01f, 0.12f, 0),
            new Vector3(0.05f, 0.24f, RoomD), DarkBrown, wallRoot.transform);
    }

    // ────────────────────────────────

    private static void CreateWindow()
    {
        var windowRoot = new GameObject("Windows") { transform = { parent = _root.transform } };

        // Large front window panes (3 panes)
        // Window area: from y=0.5 to y=2.75 (h=2.25), across front wall
        float winY = 1.625f;
        float winH = 2.25f;
        float paneW = 3.2f;
        float paneSpacing = 4.0f;

        for (int i = 0; i < 3; i++)
        {
            float x = -4f + i * paneSpacing;
            // Glass pane
            var glass = CreateQuad($"WindowGlass_{i}",
                new Vector3(x, winY, RoomD/2 - WallT - 0.02f),
                new Vector3(paneW, winH), GlassBlue, windowRoot.transform);

            // Window frame
            float frameT = 0.06f;
            CreateBox($"FrameTop_{i}", new Vector3(x, winY + winH/2, RoomD/2 - WallT/2),
                new Vector3(paneW + frameT, frameT, WallT), DarkBrown, windowRoot.transform);
            CreateBox($"FrameBottom_{i}", new Vector3(x, winY - winH/2, RoomD/2 - WallT/2),
                new Vector3(paneW + frameT, frameT, WallT), DarkBrown, windowRoot.transform);
            CreateBox($"FrameLeft_{i}", new Vector3(x - paneW/2, winY, RoomD/2 - WallT/2),
                new Vector3(frameT, winH, WallT), DarkBrown, windowRoot.transform);
            CreateBox($"FrameRight_{i}", new Vector3(x + paneW/2, winY, RoomD/2 - WallT/2),
                new Vector3(frameT, winH, WallT), DarkBrown, windowRoot.transform);

            // Window sill
            CreateBox($"Sill_{i}", new Vector3(x, 0.45f, RoomD/2 - WallT - 0.12f),
                new Vector3(paneW + 0.2f, 0.08f, 0.25f), LightWood, windowRoot.transform);
        }

        // Small side window on left wall
        var sideGlass = CreateQuad("SideWindowGlass",
            new Vector3(-RoomW/2 + WallT + 0.02f, 1.7f, 1.5f),
            new Vector3(1.5f, 1.5f), GlassBlue, windowRoot.transform);
        CreateBox("SideWindowFrame", new Vector3(-RoomW/2 + WallT/2, 1.7f, 1.5f),
            new Vector3(WallT, 1.6f, 1.6f), DarkBrown, windowRoot.transform);
    }

    // ────────────────────────────────

    private static void CreateCounter()
    {
        var counterRoot = new GameObject("Counter") { transform = { parent = _root.transform } };

        // Main counter body (back wall, slightly right of center)
        float cx = 1.5f, cz = -RoomD/2 + WallT + 1.2f;
        CreateBox("CounterBody", new Vector3(cx, 0.55f, cz),
            new Vector3(3.5f, 1.1f, 0.7f), DarkBrown, counterRoot.transform);
        // Counter top
        CreateBox("CounterTop", new Vector3(cx, 1.12f, cz),
            new Vector3(3.7f, 0.06f, 0.8f), CounterTop, counterRoot.transform);

        // Back shelf above counter
        CreateBox("CounterShelf", new Vector3(cx, 1.85f, cz - 0.25f),
            new Vector3(3.3f, 0.05f, 0.35f), LightWood, counterRoot.transform);

        // Espresso machine placeholder
        CreateBox("EspressoMachine", new Vector3(cx - 0.8f, 1.35f, cz - 0.15f),
            new Vector3(0.5f, 0.4f, 0.45f), MetalGray, counterRoot.transform);
        // Cups placeholder
        for (int i = 0; i < 3; i++)
        {
            CreateCylinder($"Cup_{i}",
                new Vector3(cx + 0.6f + i * 0.2f, 1.95f, cz - 0.2f),
                0.06f, 0.12f, White, counterRoot.transform);
        }
        // Menu board on wall
        CreateBox("MenuBoard", new Vector3(cx + 0.3f, 2.1f, -RoomD/2 + WallT + 0.08f),
            new Vector3(1.8f, 1.2f, 0.04f), DarkBrown * 1.3f, counterRoot.transform);

        // Cash register
        CreateBox("CashRegister", new Vector3(cx + 1.2f, 1.2f, cz + 0.15f),
            new Vector3(0.35f, 0.15f, 0.25f), DarkBrown, counterRoot.transform);
    }

    // ────────────────────────────────

    private static void CreateCustomerSeating()
    {
        var seatRoot = new GameObject("CustomerSeating") { transform = { parent = _root.transform } };

        // 3 tables with chairs
        CreateTableSet("TableSet1", new Vector3( 2.5f, 0,  1.5f), seatRoot.transform);
        CreateTableSet("TableSet2", new Vector3( -1f, 0,  2.5f), seatRoot.transform);
        CreateTableSet("TableSet3", new Vector3( -3f, 0, -0.5f), seatRoot.transform);

        // Window bar seating
        float barY = 1.15f;
        float barZ = RoomD/2 - WallT - 0.35f;
        CreateBox("WindowBar", new Vector3(0, barY, barZ),
            new Vector3(10f, 0.06f, 0.55f), LightWood, seatRoot.transform);

        // Bar stools
        for (int i = 0; i < 5; i++)
        {
            float sx = -4f + i * 2f;
            CreateBarStool($"BarStool_{i}", new Vector3(sx, 0, barZ - 0.4f), seatRoot.transform);
        }
    }

    private static void CreateTableSet(string name, Vector3 center, Transform parent)
    {
        var setRoot = new GameObject(name) { transform = { parent = parent, position = center } };

        // Round table
        CreateCylinder("TableTop", new Vector3(0, 0.7f, 0), 0.45f, 0.06f, LightWood, setRoot.transform);
        CreateCylinder("TableLeg", new Vector3(0, 0.35f, 0), 0.06f, 0.7f, DarkBrown, setRoot.transform);

        // 2 chairs
        CreateChair("Chair1", new Vector3( 0.6f, 0, 0), setRoot.transform);
        CreateChair("Chair2", new Vector3(-0.6f, 0, 0), setRoot.transform);
    }

    private static void CreateChair(string name, Vector3 pos, Transform parent)
    {
        var chair = new GameObject(name) { transform = { parent = parent, position = pos } };
        // Seat
        CreateBox("Seat", new Vector3(0, 0.42f, 0), new Vector3(0.4f, 0.06f, 0.4f), DarkBrown, chair.transform);
        // Back
        CreateBox("Back", new Vector3(0, 0.7f, -0.18f), new Vector3(0.36f, 0.5f, 0.04f), DarkBrown, chair.transform);
        // Legs
        float lx = 0.14f, lz = 0.14f;
        CreateCylinder("Leg_FL", new Vector3( lx, 0.21f,  lz), 0.025f, 0.42f, DarkBrown, chair.transform);
        CreateCylinder("Leg_FR", new Vector3(-lx, 0.21f,  lz), 0.025f, 0.42f, DarkBrown, chair.transform);
        CreateCylinder("Leg_BL", new Vector3( lx, 0.21f, -lz), 0.025f, 0.42f, DarkBrown, chair.transform);
        CreateCylinder("Leg_BR", new Vector3(-lx, 0.21f, -lz), 0.025f, 0.42f, DarkBrown, chair.transform);
    }

    private static void CreateBarStool(string name, Vector3 pos, Transform parent)
    {
        var stool = new GameObject(name) { transform = { parent = parent, position = pos } };
        CreateCylinder("StoolSeat", new Vector3(0, 0.88f, 0), 0.18f, 0.05f, MetalGray, stool.transform);
        CreateCylinder("StoolPole", new Vector3(0, 0.44f, 0), 0.04f, 0.88f, MetalGray, stool.transform);
        CreateCylinder("StoolBase", new Vector3(0, 0.03f, 0), 0.2f, 0.06f, MetalGray, stool.transform);
    }

    // ────────────────────────────────

    private static void CreateCatFurnitureZone()
    {
        var catZone = new GameObject("CatFurnitureZone") { transform = { parent = _root.transform } };

        // ── Large cat tree (back-left corner) ──
        var catTree = new GameObject("CatTree_Large") { transform = { parent = catZone.transform, position = new Vector3(-5f, 0, -3f) } };
        // Base platform
        CreateBox("TreeBase", new Vector3(0, 0.08f, 0), new Vector3(0.9f, 0.16f, 0.9f), WarmWood, catTree.transform);
        // Trunk
        CreateCylinder("TreeTrunk", new Vector3(0.15f, 0.55f, 0), 0.08f, 1.1f, DarkBrown, catTree.transform);
        // Mid platform
        CreateBox("TreeMidPlatform", new Vector3(0.15f, 0.9f, 0), new Vector3(0.5f, 0.06f, 0.5f), WarmWood, catTree.transform);
        // Upper trunk
        CreateCylinder("TreeTrunk2", new Vector3(-0.1f, 1.3f, 0.1f), 0.06f, 0.8f, DarkBrown, catTree.transform);
        // Top platform
        CreateBox("TreeTopPlatform", new Vector3(-0.1f, 1.55f, 0.1f), new Vector3(0.4f, 0.06f, 0.4f), WarmWood, catTree.transform);
        // Top bed (enclosed box)
        CreateBox("TreeTopBed", new Vector3(-0.1f, 1.8f, 0.1f), new Vector3(0.5f, 0.3f, 0.45f), CushionBlue, catTree.transform);
        // Scratching post (wrapped trunk)
        CreateCylinder("ScratchPost", new Vector3(0.15f, 0.5f, 0), 0.12f, 0.6f, new Color(0.5f, 0.38f, 0.25f), catTree.transform);

        // ── Cat beds (various spots) ──
        CreateCatBed("CatBed1", new Vector3(-5.5f, 0.08f, -1.5f), CushionBlue, catZone.transform);
        CreateCatBed("CatBed2", new Vector3(-3f, 0.08f, -3.8f), CushionPink, catZone.transform);
        CreateCatBed("CatBed3", new Vector3(-4.5f, 0.9f, -2.8f), new Color(0.75f, 0.7f, 0.55f), catZone.transform);

        // ── Scratching posts ──
        CreateScratchingPost("ScratchPost1", new Vector3(-5.8f, 0, -2.8f), catZone.transform);
        CreateScratchingPost("ScratchPost2", new Vector3(-3.5f, 0, -1f), catZone.transform);

        // ── Food & water station ──
        var foodStation = new GameObject("FoodStation") { transform = { parent = catZone.transform, position = new Vector3(-5.8f, 0, -0.5f) } };
        // Placemat
        CreateBox("Placemat", new Vector3(0, 0.03f, 0), new Vector3(0.6f, 0.025f, 0.35f), new Color(0.4f, 0.6f, 0.4f), foodStation.transform);
        // Food bowl
        CreateBowl("FoodBowl", new Vector3(-0.12f, 0.06f, 0), new Color(0.75f, 0.55f, 0.35f), foodStation.transform);
        // Water bowl
        CreateBowl("WaterBowl", new Vector3(0.12f, 0.06f, 0), new Color(0.45f, 0.6f, 0.75f), foodStation.transform);

        // ── Cat toys area ──
        var toyArea = new GameObject("ToyArea") { transform = { parent = catZone.transform, position = new Vector3(-4f, 0, -0.3f) } };
        // Toy box
        CreateBox("ToyBox", new Vector3(0, 0.15f, 0), new Vector3(0.5f, 0.3f, 0.4f), LightWood, toyArea.transform);
        // Ball toys
        CreateSphere("ToyBall1", new Vector3(0.2f, 0.04f, 0.3f), 0.06f, WarmOrange, toyArea.transform);
        CreateSphere("ToyBall2", new Vector3(-0.25f, 0.04f, 0.35f), 0.05f, SoftGreen, toyArea.transform);
        CreateSphere("ToyBall3", new Vector3(0.1f, 0.04f, -0.25f), 0.07f, CushionPink, toyArea.transform);
        // Feather wand toy
        var wand = CreateCylinder("WandStick", new Vector3(-0.3f, 0.06f, -0.1f), 0.02f, 0.4f, MetalGray, toyArea.transform);
        wand.transform.rotation = Quaternion.Euler(0, 0, 30f);
        CreateSphere("WandFeather", new Vector3(-0.1f, 0.18f, -0.1f), 0.05f, CushionPink, toyArea.transform);

        // ── Wall shelves for cats ──
        CreateBox("CatShelf1", new Vector3(-RoomW/2 + WallT + 0.08f, 1.9f, -1f),
            new Vector3(0.35f, 0.05f, 0.8f), LightWood, catZone.transform);
        CreateBox("CatShelf2", new Vector3(-RoomW/2 + WallT + 0.08f, 1.9f, -3f),
            new Vector3(0.35f, 0.05f, 0.8f), LightWood, catZone.transform);
    }

    private static void CreateCatBed(string name, Vector3 pos, Color color, Transform parent)
    {
        var bed = new GameObject(name) { transform = { parent = parent, position = pos } };
        // Cushion base
        CreateCylinder("Cushion", new Vector3(0, 0.04f, 0), 0.35f, 0.08f, color, bed.transform);
        // Rim
        CreateCylinder("Rim", new Vector3(0, 0.04f, 0), 0.38f, 0.04f, color * 0.8f, bed.transform);
    }

    private static void CreateScratchingPost(string name, Vector3 pos, Transform parent)
    {
        var post = new GameObject(name) { transform = { parent = parent, position = pos } };
        CreateBox("Base", new Vector3(0, 0.04f, 0), new Vector3(0.35f, 0.08f, 0.35f), LightWood, post.transform);
        CreateCylinder("Post", new Vector3(0, 0.35f, 0), 0.06f, 0.7f, new Color(0.5f, 0.38f, 0.25f), post.transform);
        CreateSphere("Top", new Vector3(0, 0.7f, 0), 0.08f, WarmWood, post.transform);
    }

    private static void CreateBowl(string name, Vector3 pos, Color color, Transform parent)
    {
        var bowl = new GameObject(name) { transform = { parent = parent, position = pos } };
        CreateCylinder("BowlBase", new Vector3(0, 0.025f, 0), 0.08f, 0.03f, MetalGray, bowl.transform);
        CreateCylinder("BowlInner", new Vector3(0, 0.04f, 0), 0.065f, 0.02f, color, bowl.transform);
    }

    // ────────────────────────────────

    private static void CreateCozyCorner()
    {
        var cozyRoot = new GameObject("CozyCorner") { transform = { parent = _root.transform } };

        // Large sofa (right side, near back)
        var sofaPos = new Vector3(4.5f, 0, -2.5f);
        // Sofa base
        CreateBox("SofaBase", new Vector3(0, 0.25f, 0.15f), new Vector3(2.4f, 0.5f, 0.8f), new Color(0.55f, 0.4f, 0.3f), sofaPos, cozyRoot.transform);
        // Sofa back
        CreateBox("SofaBack", new Vector3(0, 0.55f, -0.38f), new Vector3(2.4f, 0.55f, 0.15f), new Color(0.5f, 0.35f, 0.25f), sofaPos, cozyRoot.transform);
        // Sofa armrests
        CreateBox("SofaArmL", new Vector3(-1.15f, 0.3f, 0.15f), new Vector3(0.15f, 0.5f, 0.75f), new Color(0.5f, 0.35f, 0.25f), sofaPos, cozyRoot.transform);
        CreateBox("SofaArmR", new Vector3( 1.15f, 0.3f, 0.15f), new Vector3(0.15f, 0.5f, 0.75f), new Color(0.5f, 0.35f, 0.25f), sofaPos, cozyRoot.transform);
        // Cushions
        CreateBox("Cushion1", new Vector3(-0.5f, 0.55f, 0.1f), new Vector3(0.55f, 0.12f, 0.5f), CushionBlue, sofaPos, cozyRoot.transform);
        CreateBox("Cushion2", new Vector3( 0.5f, 0.55f, 0.1f), new Vector3(0.55f, 0.12f, 0.5f), CushionPink, sofaPos, cozyRoot.transform);

        // Coffee table in front of sofa
        CreateBox("CoffeeTable", new Vector3(sofaPos.x, 0.35f, sofaPos.z + 1f),
            new Vector3(1.2f, 0.06f, 0.7f), LightWood, cozyRoot.transform);
        CreateBox("CoffeeTableLeg1", new Vector3(sofaPos.x - 0.5f, 0.18f, sofaPos.z + 1.25f),
            new Vector3(0.06f, 0.36f, 0.06f), DarkBrown, cozyRoot.transform);
        CreateBox("CoffeeTableLeg2", new Vector3(sofaPos.x + 0.5f, 0.18f, sofaPos.z + 1.25f),
            new Vector3(0.06f, 0.36f, 0.06f), DarkBrown, cozyRoot.transform);
        CreateBox("CoffeeTableLeg3", new Vector3(sofaPos.x - 0.5f, 0.18f, sofaPos.z + 0.75f),
            new Vector3(0.06f, 0.36f, 0.06f), DarkBrown, cozyRoot.transform);
        CreateBox("CoffeeTableLeg4", new Vector3(sofaPos.x + 0.5f, 0.18f, sofaPos.z + 0.75f),
            new Vector3(0.06f, 0.36f, 0.06f), DarkBrown, cozyRoot.transform);

        // Bookshelf
        CreateBox("Bookshelf", new Vector3(sofaPos.x + 1.5f, 1.1f, sofaPos.z - 0.6f),
            new Vector3(1f, 2.2f, 0.3f), DarkBrown, cozyRoot.transform);
        // Shelf dividers
        for (int i = 0; i < 3; i++)
        {
            CreateBox($"Shelf_{i}", new Vector3(sofaPos.x + 1.5f, 0.5f + i * 0.6f, sofaPos.z - 0.6f),
                new Vector3(0.9f, 0.03f, 0.28f), LightWood, cozyRoot.transform);
        }

        // Floor lamp
        var lamp = new GameObject("FloorLamp") { transform = { parent = cozyRoot.transform, position = new Vector3(sofaPos.x - 1.6f, 0, sofaPos.z + 0.5f) } };
        CreateCylinder("LampPole", new Vector3(0, 0.8f, 0), 0.03f, 1.6f, MetalGray, lamp.transform);
        CreateCylinder("LampShade", new Vector3(0, 1.65f, 0), 0.18f, 0.35f, new Color(0.9f, 0.85f, 0.7f), lamp.transform);
    }

    // Helper for positioned objects
    private static GameObject CreateBox(string name, Vector3 localPos, Vector3 size, Color color, Vector3 parentWorldPos, Transform parent)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = name;
        go.transform.position = parentWorldPos + localPos;
        go.transform.localScale = size;
        go.transform.SetParent(parent);
        var mat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        mat.color = color;
        go.GetComponent<Renderer>().material = mat;
        return go;
    }

    // ────────────────────────────────

    private static void CreateCafeDecor()
    {
        var decorRoot = new GameObject("Decorations") { transform = { parent = _root.transform } };

        // Potted plants
        CreatePlant("Plant1", new Vector3(-5.8f, 0, 3.5f), decorRoot.transform);
        CreatePlant("Plant2", new Vector3( 5.8f, 0, 3.5f), decorRoot.transform);
        CreatePlant("Plant3", new Vector3( 0f, 0, -RoomD/2 + WallT + 0.4f), decorRoot.transform);

        // Wall art frames
        CreateFrame("Frame1", new Vector3(-2f, 2.1f, -RoomD/2 + WallT + 0.06f), new Vector3(0.7f, 0.9f, 0.03f), decorRoot.transform);
        CreateFrame("Frame2", new Vector3( 2f, 2.3f, -RoomD/2 + WallT + 0.06f), new Vector3(0.5f, 0.6f, 0.03f), decorRoot.transform);
        CreateFrame("Frame3", new Vector3(-4f, 2.0f, -RoomD/2 + WallT + 0.06f), new Vector3(0.4f, 0.5f, 0.03f), decorRoot.transform);

        // Frame on left wall
        CreateBox("Frame4", new Vector3(-RoomW/2 + WallT + 0.06f, 2.2f, -1f),
            new Vector3(0.03f, 0.6f, 0.8f), DarkBrown, decorRoot.transform);

        // Cat-themed wall clock
        CreateCylinder("Clock", new Vector3(3f, 2.5f, -RoomD/2 + WallT + 0.08f),
            0.25f, 0.04f, White, decorRoot.transform);
        CreateCylinder("ClockCenter", new Vector3(3f, 2.5f, -RoomD/2 + WallT + 0.1f),
            0.04f, 0.04f, DarkBrown, decorRoot.transform);

        // Welcome mat at entrance
        CreateBox("WelcomeMat", new Vector3(0, 0.015f, RoomD/2 - 0.4f),
            new Vector3(1f, 0.03f, 0.6f), new Color(0.5f, 0.35f, 0.2f), decorRoot.transform);
    }

    private static void CreatePlant(string name, Vector3 pos, Transform parent)
    {
        var plant = new GameObject(name) { transform = { parent = parent, position = pos } };
        // Pot
        CreateCylinder("Pot", new Vector3(0, 0.15f, 0), 0.2f, 0.3f, BrickRed, plant.transform);
        // Leaves (spheres stacked)
        CreateSphere("Leaf1", new Vector3(0, 0.35f, 0), 0.22f, SoftGreen, plant.transform);
        CreateSphere("Leaf2", new Vector3(0.1f, 0.42f, 0.05f), 0.16f, SoftGreen * 0.8f, plant.transform);
        CreateSphere("Leaf3", new Vector3(-0.08f, 0.4f, -0.06f), 0.14f, SoftGreen * 1.1f, plant.transform);
        CreateSphere("Leaf4", new Vector3(0.05f, 0.48f, -0.03f), 0.1f, SoftGreen * 0.9f, plant.transform);
    }

    private static void CreateFrame(string name, Vector3 pos, Vector3 size, Transform parent)
    {
        var go = CreateBox(name, pos, size, DarkBrown, parent);
        // Inner canvas
        CreateBox(name + "_canvas", pos + Vector3.forward * 0.005f,
            new Vector3(size.x * 0.8f, size.y * 0.8f, 0.005f), White * 0.9f, parent);
    }

    // ────────────────────────────────

    private static void CreatePlaceholderCats()
    {
        var catsRoot = new GameObject("Cats") { transform = { parent = _root.transform } };

        // ── Oreo (tuxedo cat) - near center, active ──
        var oreo = new GameObject("Cat_Oreo") { transform = { parent = catsRoot.transform, position = new Vector3(-3.5f, 0.45f, -2f) } };
        CreateCatBody(oreo.transform, Color.black, Color.white, "黑白奶牛猫");
        oreo.AddComponent<CatPlaceholder>().Init("奥利奥", "Oreo", "傲娇的奶牛猫，嘴上说不要身体很诚实");

        // ── Xiaoxue (Persian) - near cozy corner, shy ──
        var xiaoxue = new GameObject("Cat_Xiaoxue") { transform = { parent = catsRoot.transform, position = new Vector3(4f, 0.45f, -3f) } };
        CreateCatBody(xiaoxue.transform, White, new Color(0.9f, 0.85f, 0.8f), "白色波斯猫");
        xiaoxue.AddComponent<CatPlaceholder>().Init("小雪", "Xiaoxue", "胆小的波斯猫，世界好可怕但你好温暖");

        // ── Orange (orange tabby) - near food, lazy ──
        var orange = new GameObject("Cat_Orange") { transform = { parent = catsRoot.transform, position = new Vector3(-5.5f, 0.08f, -1.2f) } };
        CreateCatBody(orange.transform, WarmOrange, new Color(1f, 0.85f, 0.6f), "橘猫");
        orange.AddComponent<CatPlaceholder>().Init("橘子", "Orange", "贪吃的橘猫，吃饱睡睡饱吃");
    }

    private static void CreateCatBody(Transform parent, Color bodyColor, Color bellyColor, string description)
    {
        // Body (ellipsoid - squashed sphere)
        var body = CreateSphere("Body", new Vector3(0, 0, 0), 1f, bodyColor, parent);
        body.transform.localScale = new Vector3(0.25f, 0.2f, 0.35f);

        // Head
        var head = new GameObject("Head") { transform = { parent = parent, position = new Vector3(0, 0.22f, 0.15f) } };
        CreateSphere("HeadSphere", new Vector3(0, 0, 0), 0.18f, bodyColor, head.transform);
        // Ears
        var earL = CreateCylinder("EarL", new Vector3(-0.09f, 0.1f, 0.05f), 0.04f, 0.12f, bodyColor, head.transform);
        earL.transform.rotation = Quaternion.Euler(15f, 0, 15f);
        var earR = CreateCylinder("EarR", new Vector3( 0.09f, 0.1f, 0.05f), 0.04f, 0.12f, bodyColor, head.transform);
        earR.transform.rotation = Quaternion.Euler(15f, 0, -15f);
        // Inner ears
        var innerL = CreateCylinder("EarInnerL", new Vector3(-0.09f, 0.1f, 0.06f), 0.025f, 0.08f, CushionPink, head.transform);
        innerL.transform.rotation = Quaternion.Euler(15f, 0, 15f);
        var innerR = CreateCylinder("EarInnerR", new Vector3( 0.09f, 0.1f, 0.06f), 0.025f, 0.08f, CushionPink, head.transform);
        innerR.transform.rotation = Quaternion.Euler(15f, 0, -15f);
        // Eyes
        CreateSphere("EyeL", new Vector3(-0.06f, 0.01f, 0.16f), 0.04f, new Color(0.2f, 0.7f, 0.3f), head.transform);
        CreateSphere("EyeR", new Vector3( 0.06f, 0.01f, 0.16f), 0.04f, new Color(0.2f, 0.7f, 0.3f), head.transform);
        // Nose
        CreateSphere("Nose", new Vector3(0, -0.03f, 0.17f), 0.025f, CushionPink, head.transform);
        // Belly
        CreateSphere("Belly", new Vector3(0, -0.05f, -0.02f), 1f, bellyColor, parent);
        var bellyGo = parent.Find("Belly");
        if (bellyGo) bellyGo.transform.localScale = new Vector3(0.2f, 0.15f, 0.25f);
        // Tail
        var tail = CreateCylinder("Tail", new Vector3(0, 0.05f, -0.2f), 0.03f, 0.25f, bodyColor, parent);
        tail.transform.rotation = Quaternion.Euler(30f, 0, 20f);
        // Paws
        CreateSphere("Paw_FL", new Vector3(-0.1f, -0.12f,  0.12f), 0.05f, bellyColor, parent);
        CreateSphere("Paw_FR", new Vector3( 0.1f, -0.12f,  0.12f), 0.05f, bellyColor, parent);
        CreateSphere("Paw_BL", new Vector3(-0.1f, -0.12f, -0.12f), 0.05f, bellyColor, parent);
        CreateSphere("Paw_BR", new Vector3( 0.1f, -0.12f, -0.12f), 0.05f, bellyColor, parent);
    }

    // ────────────────────────────────

    private static void CreateEntrance()
    {
        var entranceRoot = new GameObject("Entrance") { transform = { parent = _root.transform } };

        // Door area (right side of front wall)
        CreateBox("DoorFrame", new Vector3(RoomW/2 - 0.5f, 1.2f, RoomD/2 - WallT/2),
            new Vector3(1f, 2.4f, WallT * 1.5f), DarkBrown, entranceRoot.transform);
        // Door
        CreateBox("Door", new Vector3(RoomW/2 - 0.5f, 1.2f, RoomD/2 - WallT - 0.03f),
            new Vector3(0.85f, 2.3f, 0.06f), LightWood, entranceRoot.transform);
        // Door handle
        CreateCylinder("DoorHandle", new Vector3(RoomW/2 - 0.65f, 1.25f, RoomD/2 - WallT - 0.08f),
            0.03f, 0.08f, MetalGray, entranceRoot.transform);

        // Coat rack near entrance
        var coatRack = new GameObject("CoatRack") { transform = { parent = entranceRoot.transform, position = new Vector3(RoomW/2 - 1.7f, 0, RoomD/2 - 0.8f) } };
        CreateCylinder("RackPole", new Vector3(0, 0.8f, 0), 0.04f, 1.6f, DarkBrown, coatRack.transform);
        for (int i = 0; i < 3; i++)
        {
            CreateCylinder($"RackHook_{i}", new Vector3(0.06f, 1.3f - i * 0.25f, 0.08f), 0.015f, 0.12f, MetalGray, coatRack.transform);
        }
    }

    // ────────────────────────────────

    private static void CreateCamera()
    {
        var camObj = new GameObject("Main Camera");
        camObj.transform.position = new Vector3(0, 6.5f, -3.5f);
        camObj.transform.rotation = Quaternion.Euler(55f, 0f, 0f);

        var cam = camObj.AddComponent<Camera>();
        cam.fieldOfView = 50f;

        // Add audio listener
        camObj.AddComponent<AudioListener>();

        // Add simple camera controller
        camObj.AddComponent<SimpleCameraController>();

        // Tag as MainCamera
        camObj.tag = "MainCamera";
    }

    private static void CreateGameManager()
    {
        var gmObj = new GameObject("GameManager");
        gmObj.transform.SetParent(_root.transform);
        gmObj.AddComponent<GameManager>();
    }
}
