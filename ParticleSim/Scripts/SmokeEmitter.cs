using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SmokeEmitter : MonoBehaviour{
    public bool enable = true; 

    [Space]
    public SmokeManager smokeManager;

    [Header("发射参数")]
    public float strength = 5f;                          
    public Vector3Int emitSize = new Vector3Int(1, 1, 1); 

    private Vector3Int gridPos;                           

    void Start()
    {
        if (smokeManager == null) return;
        Vector3Int gridBottomCenter = new Vector3Int(smokeManager.size.x / 2, 1, smokeManager.size.z / 2);
        transform.position = smokeManager.gridToWorldPos(gridBottomCenter, true);
    }

    void Update()
    {
        if (enable != true) return;

        //gridPos = smokeManager.worldToGridPos(transform.position) - new Vector3Int(1/2, 1/2, 1/2);
        gridPos = smokeManager.worldToGridPos(transform.position) - new Vector3Int(emitSize.x / 2, emitSize.y / 2, emitSize.z / 2);
        
        
        float fadeOutTime = 80f; 
        float duration = 20f;   
        float fadeFactor = Mathf.Clamp01(1f - (smokeManager.simulationTime - fadeOutTime) / duration);
        float adjustedStrength = strength * Mathf.Clamp01(Time.time / 800f) * fadeFactor;

        //float adjustedStrength = strength * Mathf.Clamp01(Time.time / 100f);
        for (int x = 0; x < emitSize.x; x++)
        {
            for (int y = 0; y < emitSize.y; y++)
            {
                for (int z = 0; z < emitSize.z; z++)
                {
                    Vector3Int cell = gridPos + new Vector3Int(x, y, z);    
                    if (smokeManager.isInsideGrid(cell))                    
                    {
                        smokeManager.setDensityAtPoint(cell, adjustedStrength);
                    }
                }
            }
        }
    }
}
