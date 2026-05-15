using UnityEngine;

/// <summary>
/// Core game state manager for the cat cafe demo.
/// Tracks cats, trust levels, day cycle, and customer flow.
/// </summary>
public class GameManager : MonoBehaviour
{
    public static GameManager Instance { get; private set; }

    [Header("Cats")]
    public CatPlaceholder[] cats;

    [Header("Trust Levels (0-100)")]
    [Range(0f, 100f)] public float trustOreo;
    [Range(0f, 100f)] public float trustXiaoxue;
    [Range(0f, 100f)] public float trustOrange;

    [Header("Cafe State")]
    public int dayCount = 1;
    public float gameTime;
    public int coins;
    public int reputation;

    [Header("Customers")]
    public int customersServedToday;
    public int totalCustomersServed;

    private CatCafeController _cafeController;

    private void Awake()
    {
        if (Instance != null)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        DontDestroyOnLoad(gameObject);
    }

    private void Start()
    {
        _cafeController = FindObjectOfType<CatCafeController>();
        cats = FindObjectsOfType<CatPlaceholder>();

        Debug.Log($"[GameManager] Cat Cafe day {dayCount} begins!");
        Debug.Log($"[GameManager] {cats.Length} cats in residence:");
        foreach (var c in cats)
        {
            Debug.Log($"  - {c.catNameCN} ({c.catNameEN}): {c.personality}");
        }
    }

    private void Update()
    {
        gameTime += Time.deltaTime;
    }

    public void AddTrust(string catName, float amount)
    {
        switch (catName)
        {
            case "Oreo":   trustOreo   = Mathf.Clamp(trustOreo   + amount, 0, 100); break;
            case "Xiaoxue": trustXiaoxue = Mathf.Clamp(trustXiaoxue + amount, 0, 100); break;
            case "Orange": trustOrange = Mathf.Clamp(trustOrange + amount, 0, 100); break;
        }
    }

    public CatPlaceholder GetCat(string name)
    {
        if (cats == null) return null;
        foreach (var c in cats)
        {
            if (c != null && c.catNameEN == name) return c;
        }
        return null;
    }

    public float GetTrust(string name) => name switch
    {
        "Oreo"   => trustOreo,
        "Xiaoxue" => trustXiaoxue,
        "Orange" => trustOrange,
        _ => 0f
    };

    public string GetTrustRank(float trust) => trust switch
    {
        >= 90f => "挚友",
        >= 70f => "亲密",
        >= 50f => "信任",
        >= 25f => "熟悉",
        >= 5f  => "初识",
        _      => "陌生"
    };
}
