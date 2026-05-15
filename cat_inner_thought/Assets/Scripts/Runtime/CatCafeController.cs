using UnityEngine;

/// <summary>
/// Runtime manager for the cat cafe scene — handles ambient lighting transitions,
/// customer NPC placeholder spawning, music zones, and cat interaction points.
/// </summary>
public class CatCafeController : MonoBehaviour
{
    [Header("Lighting")]
    public Light sunLight;
    public Light[] pendantLights;
    public Gradient dayToEveningGradient = new();
    [Range(0f, 1f)] public float timeOfDay = 0.35f; // 0=dawn, 0.5=noon, 1=dusk
    public float dayCycleSpeed;

    [Header("Customer Spawn Points")]
    public Transform[] seatTransforms;
    public GameObject customerPrefab;
    public int maxCustomers = 5;

    [Header("Cat Interaction Points")]
    public Transform[] catInterestPoints;

    [Header("Audio Zones")]
    public AudioSource bgmSource;
    public AudioSource ambienceSource;

    private float _cycleTimer;
    private GameObject[] _activeCustomers;

    private void Awake()
    {
        _activeCustomers = new GameObject[maxCustomers];
    }

    private void Start()
    {
        // Auto-discover lights
        var lights = FindObjectsOfType<Light>();
        foreach (var l in lights)
        {
            if (l.type == LightType.Directional) sunLight = l;
        }

        // Find pendant lights under Lighting/CatCafe_Root
        var lightRoot = transform.Find("Lighting");
        if (lightRoot != null)
        {
            var pts = lightRoot.GetComponentsInChildren<Light>();
            var list = new System.Collections.Generic.List<Light>();
            foreach (var l in pts)
                if (l.type == LightType.Point) list.Add(l);
            pendantLights = list.ToArray();
        }

        // Gather cat interest points from cat zone
        var catZone = transform.Find("CatZone");
        if (catZone != null)
        {
            var children = new System.Collections.Generic.List<Transform>();
            foreach (Transform t in catZone.GetComponentsInChildren<Transform>())
            {
                if (t.name.Contains("Bed") || t.name.Contains("Scratch") ||
                    t.name.Contains("Tree") || t.name.Contains("Station") ||
                    t.name.Contains("Hammock") || t.name.Contains("Shelf"))
                {
                    children.Add(t);
                }
            }
            catInterestPoints = children.ToArray();
        }

        ApplyTimeOfDay();
    }

    private void Update()
    {
        if (dayCycleSpeed > 0f)
        {
            _cycleTimer += Time.deltaTime * dayCycleSpeed;
            timeOfDay = (Mathf.Sin(_cycleTimer) + 1f) / 2f;
            ApplyTimeOfDay();
        }
    }

    private void ApplyTimeOfDay()
    {
        if (sunLight != null)
        {
            sunLight.color = dayToEveningGradient.Evaluate(timeOfDay);
            sunLight.intensity = Mathf.Lerp(1.5f, 0.3f, timeOfDay);
            sunLight.transform.rotation = Quaternion.Euler(
                Mathf.Lerp(20f, 70f, timeOfDay),
                -30f, 0f
            );
        }

        if (pendantLights != null)
        {
            foreach (var l in pendantLights)
            {
                l.intensity = Mathf.Lerp(0.5f, 3.0f, timeOfDay);
            }
        }

        if (ambienceSource != null)
        {
            ambienceSource.volume = Mathf.Lerp(0.4f, 0.8f, timeOfDay);
        }
    }

    public Transform GetNearestInterestPoint(Vector3 from, float maxRange = 10f)
    {
        Transform best = null;
        float bestDist = maxRange;
        if (catInterestPoints == null) return null;

        foreach (var t in catInterestPoints)
        {
            if (t == null) continue;
            float d = Vector3.Distance(from, t.position);
            if (d < bestDist)
            {
                bestDist = d;
                best = t;
            }
        }
        return best;
    }

    private void OnDrawGizmosSelected()
    {
        if (catInterestPoints != null)
        {
            Gizmos.color = Color.yellow;
            foreach (var t in catInterestPoints)
            {
                if (t != null) Gizmos.DrawWireSphere(t.position, 0.2f);
            }
        }
    }
}
