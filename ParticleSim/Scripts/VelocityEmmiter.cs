using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VelocityEmmiter : MonoBehaviour {
    public bool enable = true;
    [Space]
    public SmokeManager smokeManager;
    public Vector3 velocity = new Vector3(0,5,0);
    //public Vector3Int size = Vector3Int.one;
    private Vector3Int gridPos;
    public Vector3Int size = new Vector3Int(1, 1, 1); 
    public float noiseScale = 0.5f;
    
    void Start()
    {
        if (smokeManager == null) return;
        Vector3Int gridBottomCenter = new Vector3Int(smokeManager.size.x / 2, 1, smokeManager.size.z / 2);
        transform.position = smokeManager.gridToWorldPos(gridBottomCenter, true);
    }

    void Update()
    {
        if (enable)
        {
            gridPos = smokeManager.worldToGridPos(transform.position) - new Vector3Int(size.x / 2, size.y / 2, size.z / 2);

            for (int x = 0; x < size.x; ++x)
            {
                for (int y = 0; y < size.y; ++y)
                {
                    for (int z = 0; z < size.z; ++z)
                    {
                        Vector3Int cell = gridPos + new Vector3Int(x, y, z);    
                        if (smokeManager.isInsideGrid(cell))                   
                        {
                            float nx = Mathf.PerlinNoise(x * 0.1f, Time.time * 0.1f);
                            float ny = Mathf.PerlinNoise(y * 0.1f, Time.time * 0.2f);
                            float nz = Mathf.PerlinNoise(z * 0.1f, Time.time * 0.3f);
                            Vector3 noise = new Vector3(nx, ny, nz) * noiseScale;
                            Vector3 finalVelocity = velocity + noise;
                            smokeManager.setVelocityAtPoint(cell, finalVelocity);
                        }
                    }
                }
            }
        }
    }
}
