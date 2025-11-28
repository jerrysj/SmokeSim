using UnityEngine;

[ExecuteAlways]
public class LidarConfigurator : MonoBehaviour
{
    public Transform targetSmokeCenter;
    public float horizontalDistance = 5f;
    public float vfovTopDeg = 2.4f;
    public float vfovBottomDeg = -17.6f;

    void Update()
    {
        if (targetSmokeCenter == null) return;

        float bottomAngleRad = Mathf.Abs(vfovBottomDeg) * Mathf.Deg2Rad;

        float requiredHeight = Mathf.Tan(bottomAngleRad) * horizontalDistance;

        Vector3 center = targetSmokeCenter.position;
        //transform.position = new Vector3(center.x, requiredHeight+1, center.z - horizontalDistance);
        transform.position = new Vector3(center.x - horizontalDistance, requiredHeight - 1f, center.z);

        transform.rotation = Quaternion.Euler(0f, 0f, 0f);
    }


#if UNITY_EDITOR
    void OnDrawGizmos()
    {
        if (targetSmokeCenter == null) return;

        Gizmos.color = Color.green;
        Gizmos.DrawLine(transform.position, targetSmokeCenter.position);

        float maxRay = 10f;
        Vector3 forward = transform.right;

        Vector3 topRay = Quaternion.Euler(0, 0, vfovTopDeg) * forward;
        Vector3 bottomRay = Quaternion.Euler(0, 0, vfovBottomDeg) * forward;

        Gizmos.color = Color.yellow;
        Gizmos.DrawRay(transform.position, topRay * maxRay);

        Gizmos.color = Color.red;
        Gizmos.DrawRay(transform.position, bottomRay * maxRay);
    }
#endif
}

