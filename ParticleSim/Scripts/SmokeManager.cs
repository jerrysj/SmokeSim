using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using System.Threading;


public class SmokeManager : MonoBehaviour {
    // solver variables
    public Vector3Int size = new Vector3Int(32, 32, 32); 
    public float scale = 20f;                             
    
    //public FluidSolver fs = new FluidSolver();              
    public FluidSolver fs;                   
    public float dt = 0.4f;                                      
    public float simulationTime = 0.0f;                  
    private Thread calcThread;                           
    public Transform obj;                                 

    private void Start()
    {
        EnsureGridAboveGround();  
        Setup();                                 
        // SetupShader();                            
        calcThread = new Thread(CalculateSmoke);  
        calcThread.Start();                      
    }

   
    void OnValidate()
    {
        if (Application.isPlaying == false)
            EnsureGridAboveGround();
    }
    
    private void EnsureGridAboveGround()
    {
        Vector3 worldBottom = gridToWorldPos(Vector3Int.zero, true);  

        if (worldBottom.y < 0f)
        {
            float offsetY = -worldBottom.y + 0.5f; 
            transform.position += new Vector3(0, offsetY, 0);
        }
    }
    

    private void Update()
    {
        // material.SetMatrix("_WorldToObject", obj.worldToLocalMatrix);  
        if (calcThread.ThreadState != ThreadState.Running)   
        {
            //Display();                                      
            //onSmokeUpdate.Invoke();                         
            calcThread = new Thread(CalculateSmoke);         
            calcThread.Start();                             
        }
    }

    
    // Constructor
    public void Setup()
    {
        this.fs.setup(this.size - new Vector3Int(2, 2, 2), this.dt);   
        this.CalculateSmoke();                                       
    }

   
    public void CalculateSmoke()
    {
        this.fs.velocitySolver();                   
        this.fs.densitySolver();                    
        simulationTime += fs.dt;                   
    }

    
    public float gridScale
    {
        get { return ((float)(size.x) / scale); }   
    }

  
    public Vector3Int worldToGridPos(Vector3 pos)
    {
       
        Vector3 localPos = this.obj.transform.worldToLocalMatrix.MultiplyPoint(pos);  
        localPos *= this.gridScale;                                                  
        localPos += ((Vector3)this.size * 0.5f);                                     
        return new Vector3Int((int)localPos.x, (int)localPos.y, (int)localPos.z);     
    }

    public Vector3 gridToWorldPos(Vector3Int pos, bool inCenter = false)
    {
        Vector3 localPos = pos;
        localPos -= ((Vector3)this.size * 0.5f);                                
        if (inCenter) localPos += new Vector3(0.5f, 0.5f, 0.5f);                
        localPos /= this.gridScale;                                             
        return this.obj.transform.localToWorldMatrix.MultiplyPoint(localPos);   
    }

   
    public bool isInsideGrid(Vector3Int gridPos)
    {
        return (gridPos.x >= 0 && gridPos.x < this.size.x &&
                gridPos.y >= 0 && gridPos.y < this.size.y &&
                gridPos.z >= 0 && gridPos.z < this.size.z);
    }

    
    public void addDensityAtPoint(Vector3Int position, float dens)
    {
        fs.d[fs.id(position)] += dens;  
    }

    public void setDensityAtPoint(Vector3Int position, float dens)
    {
        fs.d[fs.id(position)] = dens;  
    }

  
    public void setVelocityAtPoint(Vector3Int position, Vector3 vel)
    {
        fs.u[fs.id(position)] = vel.x;  
        fs.v[fs.id(position)] = vel.y;  
        fs.w[fs.id(position)] = vel.z;  
    }

    void OnDrawGizmosSelected()
    {
        Gizmos.DrawWireCube(obj.position, new Vector3(size.x / gridScale, size.y / gridScale, size.z / gridScale));
    }
}
