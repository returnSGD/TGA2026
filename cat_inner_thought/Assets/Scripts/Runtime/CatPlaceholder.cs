using UnityEngine;

/// <summary>
/// Placeholder cat behavior — idle animation and floating name label.
/// Attached to each placeholder cat GameObject by CatCafeSceneBuilder.
/// </summary>
public class CatPlaceholder : MonoBehaviour
{
    public string catNameCN;
    public string catNameEN;
    public string personality;

    private float _idleTimer;
    private Vector3 _basePos;
    private Transform _head;

    public void Init(string nameCN, string nameEN, string personalityDesc)
    {
        catNameCN = nameCN;
        catNameEN = nameEN;
        personality = personalityDesc;
        gameObject.name = $"Cat_{nameEN}";
    }

    private void Start()
    {
        _basePos = transform.position;
        _head = transform.Find("Head");
        _idleTimer = Random.Range(0f, 3f);
    }

    private void Update()
    {
        _idleTimer -= Time.deltaTime;

        // Gentle idle breathing animation
        float breathe = 1f + Mathf.Sin(Time.time * 2f) * 0.02f;
        transform.localScale = Vector3.one * breathe;

        // Occasional head turn
        if (_head != null)
        {
            float headTurn = Mathf.Sin(Time.time * 1.5f + GetInstanceID() * 0.1f) * 20f;
            _head.localRotation = Quaternion.Euler(0f, headTurn, 0f);
        }

        // Very slight random movement
        if (_idleTimer <= 0f)
        {
            _idleTimer = Random.Range(2f, 5f);
            _basePos += new Vector3(
                Random.Range(-0.3f, 0.3f),
                0f,
                Random.Range(-0.3f, 0.3f)
            );
            // Clamp to room
            _basePos.x = Mathf.Clamp(_basePos.x, -6.5f, 6.5f);
            _basePos.z = Mathf.Clamp(_basePos.z, -4.5f, 4.5f);
        }

        transform.position = Vector3.Lerp(transform.position, _basePos, Time.deltaTime * 0.5f);
    }

    private void OnGUI()
    {
        // Show name label above cat
        var screenPos = Camera.main.WorldToScreenPoint(transform.position + Vector3.up * 0.5f);
        if (screenPos.z > 0)
        {
            var style = new GUIStyle(GUI.skin.label);
            style.fontSize = 14;
            style.alignment = TextAnchor.MiddleCenter;
            style.normal.textColor = Color.white;
            var labelRect = new Rect(screenPos.x - 50, Screen.height - screenPos.y - 20, 100, 20);
            GUI.Label(labelRect, catNameCN, style);

            // Personality hint in smaller text
            style.fontSize = 10;
            style.normal.textColor = new Color(1f, 1f, 1f, 0.7f);
            var hintRect = new Rect(screenPos.x - 60, Screen.height - screenPos.y, 120, 20);
            GUI.Label(hintRect, personality, style);
        }
    }
}
