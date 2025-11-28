using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class GridDebugDrawer : MonoBehaviour
{
    public SmokeManager smokeManager;
    public FluidSolver fluidSolver;

    [Header("密度可视化")]
    public bool drawDensityCells = true;
    public float densityThreshold = 0.01f;
    public float cubeSize = 0.2f;

    [Header("网格显示")]
    public bool drawGridBounds = true;
    public Color gridBoundsColor = Color.cyan;

    [Header("坐标标注")]
    public bool drawCellCoordinates = true;
    public Color labelColor = Color.yellow;
    public float labelScale = 0.5f;

    [Header("网格中心显示")]
    public bool showGridCenter = true;
    public Color gridCenterColor = Color.green;

    void OnDrawGizmos()
    {
        if (!Application.isPlaying || smokeManager == null || fluidSolver == null || fluidSolver.d == null)
            return;

        Vector3Int gridSize = fluidSolver.size;

        for (int x = 1; x <= gridSize.x; x++)
        {
            for (int y = 1; y <= gridSize.y; y++)
            {
                for (int z = 1; z <= gridSize.z; z++)
                {
                    int idx = fluidSolver.id(x, y, z);
                    float density = fluidSolver.d[idx];

                    if (drawDensityCells && density > densityThreshold)
                    {
                        Vector3Int gridPos = new Vector3Int(x, y, z);
                        Vector3 worldPos = smokeManager.gridToWorldPos(gridPos, true);

                        // 颜色映射（白到红）
                        Color densityColor = Color.Lerp(Color.white, Color.red, Mathf.Clamp01(density));
                        Gizmos.color = densityColor;
                        Gizmos.DrawCube(worldPos, Vector3.one * cubeSize);

#if UNITY_EDITOR
                        // 画坐标文本
                        if (drawCellCoordinates)
                        {
                            GUIStyle style = new GUIStyle();
                            style.normal.textColor = labelColor;
                            style.fontSize = 10;
                            Handles.Label(worldPos + Vector3.up * 0.1f, $"({x},{y},{z})", style);
                        }
#endif
                    }
                }
            }
        }

        // 绘制网格中心
        if (showGridCenter)
        {
            Vector3 worldCenter = smokeManager.gridToWorldPos(gridSize / 2, true);
            Gizmos.color = gridCenterColor;
            Gizmos.DrawWireSphere(worldCenter, 0.5f);
        }

        // 绘制整个模拟网格边界框
        if (drawGridBounds)
        {
            Gizmos.color = gridBoundsColor;
            Vector3 worldMin = smokeManager.gridToWorldPos(Vector3Int.one, true);
            Vector3 worldMax = smokeManager.gridToWorldPos(gridSize, true);
            Vector3 center = (worldMin + worldMax) * 0.5f;
            Vector3 size = new Vector3(
                Mathf.Abs(worldMax.x - worldMin.x),
                Mathf.Abs(worldMax.y - worldMin.y),
                Mathf.Abs(worldMax.z - worldMin.z)
            );
            Gizmos.DrawWireCube(center, size);
        }
    }
}
