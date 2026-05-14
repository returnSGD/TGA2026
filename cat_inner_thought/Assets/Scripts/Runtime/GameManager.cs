using UnityEngine;

/// <summary>
/// Simple game manager for the cat cafe demo.
/// Tracks the three cats and provides basic game state.
/// </summary>
public class GameManager : MonoBehaviour
{
    public static GameManager Instance { get; private set; }

    [Header("Cats")]
    public CatPlaceholder[] cats;

    [Header("Game State")]
    public float gameTime;
    public int dayCount = 1;
    public float trustOreo;
    public float trustXiaoxue;
    public float trustOrange;

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
        cats = FindObjectsOfType<CatPlaceholder>();
        Debug.Log($"[GameManager] Found {cats.Length} cats in the cafe.");
        foreach (var c in cats)
        {
            Debug.Log($"  - {c.catNameCN} ({c.catNameEN}): {c.personality}");
        }
    }

    private void Update()
    {
        gameTime += Time.deltaTime;
    }
}
