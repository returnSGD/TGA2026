using UnityEngine;

/// <summary>
/// Placeholder cat with idle behavior, name label, and basic AI state.
/// Attached by CatCafeSceneBuilder to each cat GameObject.
/// </summary>
public class CatPlaceholder : MonoBehaviour
{
    public string catNameCN;
    public string catNameEN;
    public string personality;

    public enum CatState { Idle, Sleeping, Walking, Playing, Eating, Grooming }

    [Header("State")]
    public CatState currentState = CatState.Idle;

    private float _stateTimer;
    private float _idleActionTimer;
    private Vector3 _moveTarget;
    private Vector3 _basePosition;
    private Transform _head;
    private Transform _tail;
    private CatCafeController _controller;
    private float _breatheOffset;

    public void Init(string nameCN, string nameEN, string personalityDesc)
    {
        catNameCN = nameCN;
        catNameEN = nameEN;
        personality = personalityDesc;
        gameObject.name = $"Cat_{nameEN}";
    }

    private void Start()
    {
        _basePosition = transform.position;
        _head = transform.Find("Head");
        _tail = transform.Find("Tail");
        _controller = FindObjectOfType<CatCafeController>();
        _breatheOffset = Random.Range(0f, Mathf.PI * 2f);
        _stateTimer = Random.Range(3f, 8f);
        PickNewState();
    }

    private void Update()
    {
        _stateTimer -= Time.deltaTime;
        if (_stateTimer <= 0f)
        {
            PickNewState();
            _stateTimer = Random.Range(3f, 10f);
        }

        // Breathing animation
        float breathe = 1f + Mathf.Sin(Time.time * 2f + _breatheOffset) * 0.025f;
        transform.localScale = Vector3.one * breathe;

        // Head turning
        if (_head != null)
        {
            float turn = Mathf.Sin(Time.time * 1.3f + _breatheOffset) * 25f;
            _head.localRotation = Quaternion.Slerp(_head.localRotation,
                Quaternion.Euler(0f, turn, 0f), Time.deltaTime * 2f);
        }

        // Tail wagging
        if (_tail != null)
        {
            float wag = Mathf.Sin(Time.time * 3f + _breatheOffset + 1f) * 15f;
            _tail.localRotation = Quaternion.Slerp(_tail.localRotation,
                Quaternion.Euler(wag, 0f, 20f), Time.deltaTime * 3f);
        }

        // State behavior
        switch (currentState)
        {
            case CatState.Walking:
                WalkUpdate();
                break;
            case CatState.Playing:
                PlayUpdate();
                break;
            case CatState.Sleeping:
                SleepUpdate();
                break;
            default:
                IdleUpdate();
                break;
        }
    }

    private void PickNewState()
    {
        float roll = Random.value;
        if (roll < 0.35f)       SetState(CatState.Idle);
        else if (roll < 0.55f)  SetState(CatState.Sleeping);
        else if (roll < 0.75f)  SetState(CatState.Walking);
        else if (roll < 0.90f)  SetState(CatState.Grooming);
        else                     SetState(CatState.Playing);
    }

    private void SetState(CatState state)
    {
        currentState = state;
        switch (state)
        {
            case CatState.Walking:
                PickWalkTarget();
                break;
            case CatState.Playing:
                if (_controller != null)
                {
                    var pt = _controller.GetNearestInterestPoint(transform.position, 6f);
                    if (pt != null) _moveTarget = pt.position;
                }
                break;
        }
    }

    private void PickWalkTarget()
    {
        var offset = new Vector3(
            Random.Range(-3f, 3f), 0f, Random.Range(-3f, 3f));
        _moveTarget = _basePosition + offset;
        _moveTarget.x = Mathf.Clamp(_moveTarget.x, -6.5f, 6.5f);
        _moveTarget.z = Mathf.Clamp(_moveTarget.z, -4.5f, 4.5f);
    }

    private void WalkUpdate()
    {
        Vector3 target = _moveTarget;
        target.y = transform.position.y;
        float dist = Vector3.Distance(transform.position, target);
        if (dist < 0.1f)
        {
            PickWalkTarget();
        }
        transform.position = Vector3.MoveTowards(transform.position, target, Time.deltaTime * 0.8f);
        // Face movement direction
        var dir = (target - transform.position).normalized;
        if (dir.magnitude > 0.01f)
        {
            var lookRot = Quaternion.LookRotation(dir);
            transform.rotation = Quaternion.Slerp(transform.rotation, lookRot, Time.deltaTime * 4f);
        }
    }

    private void PlayUpdate()
    {
        float dist = Vector3.Distance(transform.position, _moveTarget);
        if (dist < 0.3f)
        {
            // Bat at toy
            float bob = Mathf.Abs(Mathf.Sin(Time.time * 5f)) * 0.08f;
            transform.position += Vector3.up * bob * Time.deltaTime * 3f;
        }
        else
        {
            WalkUpdate();
        }
    }

    private void SleepUpdate()
    {
        // Gentle breathing only — no movement
        float sleepBreath = 1f + Mathf.Sin(Time.time * 1.2f + _breatheOffset) * 0.04f;
        transform.localScale = new Vector3(transform.localScale.x, sleepBreath * 0.7f, transform.localScale.z);
    }

    private void IdleUpdate()
    {
        _idleActionTimer -= Time.deltaTime;
        if (_idleActionTimer <= 0f)
        {
            _idleActionTimer = Random.Range(2f, 5f);
            _basePosition += new Vector3(Random.Range(-0.5f, 0.5f), 0f, Random.Range(-0.5f, 0.5f));
            _basePosition.x = Mathf.Clamp(_basePosition.x, -6.5f, 6.5f);
            _basePosition.z = Mathf.Clamp(_basePosition.z, -4.5f, 4.5f);
        }
        transform.position = Vector3.Lerp(transform.position, _basePosition, Time.deltaTime * 0.4f);
    }

    private void OnGUI()
    {
        var cam = Camera.main;
        if (cam == null) return;

        var worldPos = transform.position + Vector3.up * 0.55f;
        var screenPos = cam.WorldToScreenPoint(worldPos);
        if (screenPos.z <= 0) return;

        float sx = screenPos.x;
        float sy = Screen.height - screenPos.y;

        // Name label background
        var nameStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 15,
            alignment = TextAnchor.MiddleCenter,
            normal = { textColor = Color.white }
        };
        var bgStyle = new GUIStyle(GUI.skin.box)
        {
            normal = { background = MakeTex(2, 2, new Color(0, 0, 0, 0.55f)) }
        };

        float labelW = 100f, labelH = 24f;
        GUI.Box(new Rect(sx - labelW / 2, sy - labelH - 5, labelW, labelH + 18), "", bgStyle);
        GUI.Label(new Rect(sx - labelW / 2, sy - labelH - 5, labelW, labelH), catNameCN, nameStyle);

        // Personality hint
        var hintStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 10,
            alignment = TextAnchor.MiddleCenter,
            normal = { textColor = new Color(1f, 1f, 1f, 0.65f) }
        };
        GUI.Label(new Rect(sx - 60, sy + 12, 120, 18), personality, hintStyle);
    }

    private static Texture2D MakeTex(int w, int h, Color col)
    {
        var pix = new Color[w * h];
        for (int i = 0; i < pix.Length; i++) pix[i] = col;
        var tex = new Texture2D(w, h);
        tex.SetPixels(pix);
        tex.Apply();
        return tex;
    }
}
