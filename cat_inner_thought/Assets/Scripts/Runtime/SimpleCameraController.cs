using UnityEngine;

/// <summary>
/// Simple mouse + keyboard camera controller for the cat cafe demo.
/// Right-click drag to orbit, scroll to zoom, WASD to pan.
/// </summary>
public class SimpleCameraController : MonoBehaviour
{
    [Header("Orbit")]
    public float orbitSpeed = 3f;
    public float minPitch = 20f;
    public float maxPitch = 80f;

    [Header("Zoom")]
    public float zoomSpeed = 5f;
    public float minDistance = 3f;
    public float maxDistance = 20f;

    [Header("Pan")]
    public float panSpeed = 8f;

    private Vector3 _orbitTarget = new Vector3(0f, 1.2f, 0f);
    private float _yaw;
    private float _pitch = 55f;
    private float _distance = 9f;

    private void Start()
    {
        var dir = transform.position - _orbitTarget;
        _distance = dir.magnitude;
        _yaw = Mathf.Atan2(dir.x, dir.z) * Mathf.Rad2Deg;
        _pitch = Mathf.Asin(dir.y / _distance) * Mathf.Rad2Deg;
    }

    private void Update()
    {
        // Right-click orbit
        if (Input.GetMouseButton(1))
        {
            _yaw   += Input.GetAxis("Mouse X") * orbitSpeed;
            _pitch -= Input.GetAxis("Mouse Y") * orbitSpeed;
            _pitch  = Mathf.Clamp(_pitch, minPitch, maxPitch);
        }

        // Scroll zoom
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        _distance -= scroll * zoomSpeed;
        _distance  = Mathf.Clamp(_distance, minDistance, maxDistance);

        // WASD pan (shift + drag also pans)
        var pan = Vector3.zero;
        if (Input.GetKey(KeyCode.W)) pan += Vector3.forward;
        if (Input.GetKey(KeyCode.S)) pan += Vector3.back;
        if (Input.GetKey(KeyCode.A)) pan += Vector3.left;
        if (Input.GetKey(KeyCode.D)) pan += Vector3.right;
        if (Input.GetMouseButton(2))
        {
            pan += Vector3.left   * Input.GetAxis("Mouse X") * 0.3f;
            pan += Vector3.back   * Input.GetAxis("Mouse Y") * 0.3f;
        }

        _orbitTarget += transform.TransformDirection(pan) * panSpeed * Time.deltaTime;

        // Apply position
        var rot = Quaternion.Euler(_pitch, _yaw, 0f);
        var offset = rot * (Vector3.forward * -_distance);
        transform.position = _orbitTarget - offset;
        transform.LookAt(_orbitTarget);
    }

    private void OnGUI()
    {
        GUILayout.BeginArea(new Rect(10, 10, 300, 120));
        GUILayout.Box("Right-drag: Orbit | Middle-drag/WASD: Pan | Scroll: Zoom", GUILayout.ExpandWidth(true));
        GUILayout.EndArea();
    }
}
