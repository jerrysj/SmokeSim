using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;


public class FluidSolver : MonoBehaviour{
    [HideInInspector]
    public Vector3Int size, bSize;  
    [HideInInspector]
    public float dt;                

    [HideInInspector]
    public float[] d;               
    [HideInInspector]
    public float[] dOld;           

    [HideInInspector]
    public float[] u, uOld;        
    [HideInInspector]
    public float[] v, vOld;         
    [HideInInspector]
    public float[] w, wOld;         
    [HideInInspector]
    public float[] curl;            
    [HideInInspector]
    public float[] temp;            
    
    public float visc = 0.001f;      
    //[HideInInspector]
    public float diff = 0.001f;      

    float i, j, k;                 
    [HideInInspector]
    public float[] tmp;             


    public void setup(Vector3Int size, float dt)
    {
        this.size = size;
        this.dt = dt;
        this.bSize = this.size + new Vector3Int(2, 2, 2); 
        
        //this.size = this.size + new Vector3Int(2, 2, 2); 

        this.reset(); 
    }

    
    public void reset()
    {
        this.d = new float[this.bSize.x * this.bSize.y * this.bSize.z];
        this.dOld = new float[this.bSize.x * this.bSize.y * this.bSize.z];
        this.u = new float[this.bSize.x * this.bSize.y * this.bSize.z];
        this.uOld = new float[this.bSize.x * this.bSize.y * this.bSize.z];
        this.v = new float[this.bSize.x * this.bSize.y * this.bSize.z];
        this.vOld = new float[this.bSize.x * this.bSize.y * this.bSize.z];
        this.w = new float[this.bSize.x * this.bSize.y * this.bSize.z];
        this.wOld = new float[this.bSize.x * this.bSize.y * this.bSize.z];
        this.curl = new float[this.bSize.x * this.bSize.y * this.bSize.z];
        this.temp = new float[this.bSize.x * this.bSize.y * this.bSize.z];

        for (int i = 0; i < this.bSize.x; i++)
        {
            for (int j = 0; j < this.bSize.y; j++)
            {
                for (int k = 0; k < this.bSize.z; k++)
                {
                    this.u[id(i, j, k)] = this.uOld[id(i, j, k)] = 0.0f;
                    this.v[id(i, j, k)] = this.vOld[id(i, j, k)] = 0.0f;
                    this.w[id(i, j, k)] = this.wOld[id(i, j, k)] = 0.0f;
                    this.d[id(i, j, k)] = this.dOld[id(i, j, k)] = this.curl[id(i, j, k)] = 0.0f;
                    this.temp[id(i, j, k)] = 0.0f;
                }
            }
        }
    }


    public int id(int i, int j, int k)
    {
        return (i + this.bSize.x * j + this.bSize.x * this.bSize.y * k);  // index = i + (Nx+2)·j + (Nx+2)(Ny+2)·k
    }

    public int id(Vector3Int pos)
    {
        return (pos.x + this.bSize.x * pos.y + this.bSize.x * this.bSize.y * pos.z);
    }


    ////////////////////
    public void addSource(float[] x, float[] x0)
    {
        for (int i = 0; i < this.bSize.x; i++)
        {
            for (int j = 0; j < this.bSize.y; j++)
            {
                for (int k = 0; k < this.bSize.z; k++)
                {
                    x[id(i, j, k)] += this.dt * x0[id(i, j, k)];
                }
            }
        }
    }

    
    public float Curl(int x, int y, int z) 
    {
        float dw_dy = (this.w[id(x, y + 1, z)] - this.w[id(x, y - 1, z)]) * 0.5f;
        float dv_dz = (this.v[id(x, y, z + 1)] - this.v[id(x, y, z - 1)]) * 0.5f;
        float dw_dx = (this.w[id(x + 1, y, z)] - this.w[id(x - 1, y, z)]) * 0.5f;
        float du_dz = (this.u[id(x, y, z + 1)] - this.u[id(x, y, z - 1)]) * 0.5f;
        float du_dy = (this.u[id(x, y + 1, z)] - this.u[id(x, y - 1, z)]) * 0.5f;
        float dv_dx = (this.v[id(x + 1, y, z)] - this.v[id(x - 1, y, z)]) * 0.5f;

        this.i = dw_dy - dv_dz;     
        this.j = -(dw_dx - du_dz);  
        this.k = dv_dx - du_dy;    

        return (float)Mathf.Sqrt(this.i * this.i + this.j * this.j + this.k * this.k);
    }


    public void vorticityConfinement(float[] Fvc_x, float[] Fvc_y, float[] Fvc_z)
    {
        float dw_dx, dw_dy, dw_dz;
        float length;
        const float epsilon = 1f;    
        const float h = 1f;           

        // Calculate magnitude of curl(u,v,w) for each cell. (|w|)
        for (int x = 1; x <= this.size.x; x++)
        {
            for (int y = 1; y <= this.size.y; y++)
            {
                for (int z = 1; z <= this.size.z; z++)
                {
                    this.curl[id(x, y, z)] = this.Curl(x, y, z);  
                }
            }
        }

        for (int x = 2; x < this.size.x; x++)
        {
            for (int y = 2; y < this.size.y; y++)
            {
                for (int z = 2; z < this.size.z; z++)
                {
                    // Find derivative of the magnitude (n = del |w|)
                    dw_dx = (this.curl[id(x + 1, y, z)] - this.curl[id(x - 1, y, z)]) * 0.5f;
                    dw_dy = (this.curl[id(x, y + 1, z)] - this.curl[id(x, y - 1, z)]) * 0.5f;
                    dw_dz = (this.curl[id(x, y, z + 1)] - this.curl[id(x, y, z - 1)]) * 0.5f;

                    // Calculate vector length. (|n|): |n|=|∇|ω||
                    // Add small factor to prevent divide by zeros.
                    length = (float)Mathf.Sqrt(dw_dx * dw_dx + dw_dy * dw_dy + dw_dz * dw_dz) + 0.000001f;

                    // N = ( n/|n| ) = ∇|ω| / |∇|ω||
                    dw_dx /= length;
                    dw_dy /= length;
                    dw_dz /= length;
                   
                    // Will update the global variable i, j, k ans set them as w
                    this.Curl(x, y, z);

                    // N x w
                    Fvc_x[id(x, y, z)] = (dw_dy * this.k - dw_dz * this.j) * epsilon * h; // a2b3-a3b2
                    Fvc_y[id(x, y, z)] = -(dw_dx * this.k - dw_dz * this.i) * epsilon * h; // a3b1-a1b3
                    Fvc_z[id(x, y, z)] = (dw_dx * this.j - dw_dy * this.i) * epsilon * h; // a1b2-a2b1
                }
            }
        }
    }

    
    // -a*d*Y + b*(T-Tamb)*Y
    public void buoyancy(float[] Fbuoy)
    {
        float Tamb = 0;              
        const float a = 0.000625f;   
        const float b = 0.025f;      

        // sum all temperatures
        for (int i = 1; i <= this.size.x; i++)
        {
            for (int j = 1; j <= this.size.y; j++)
            {
                for (int k = 1; k <= this.size.z; k++)
                {
                    Tamb += this.d[id(i, j, k)];
                }
            }
        }

        // get average temperature
        Tamb /= (this.size.x * this.size.y * this.size.z);

        // for each cell compute buoyancy force
        for (int i = 1; i <= this.size.x; i++)
        {
            for (int j = 1; j <= this.size.y; j++)
            {
                for (int k = 1; k <= this.size.z; k++)
                {                   
                    Fbuoy[id(i, j, k)] = a * this.d[id(i, j, k)] + -b * (this.d[id(i, j, k)] - Tamb); 
                }
            }
        }
    }


    // util array swapping methods
    public void swapU()
    {
        this.tmp = this.u;
        this.u = this.uOld;
        this.uOld = this.tmp;
    }

    public void swapV()
    {
        this.tmp = this.v;
        this.v = this.vOld;
        this.vOld = this.tmp;
    }

    public void swapW()
    {
        this.tmp = this.w;
        this.w = this.wOld;
        this.wOld = this.tmp;
    }

    public void swapD()
    {
        this.tmp = this.d;
        this.d = this.dOld;
        this.dOld = this.tmp;
    }


    /*
    * b - flag, which tells what boundries should be held
    * c - array to store results of diffusion computation
    * c0 - The input array on which we should compute diffusion.
    * diff - the factor of diffusion
    **/
    public void diffuse(int b, float[] c, float[] c0, float diff)
    {
        float a = this.dt * diff * this.size.x * this.size.y * this.size.z;
        this.linearSolver(b, c, c0, a, 1 + 6 * a);                              // 调用线性求解器
    }

    void linearSolver(int b, float[] x, float[] x0, float a, float c)
    {
        for (int n = 0; n < 20; n++)
        {
            for (int i = 1; i <= this.size.x; i++)
            {
                for (int j = 1; j <= this.size.y; j++)
                {
                    for (int k = 1; k <= this.size.z; k++)
                    {
                        x[id(i, j, k)] = ( a * ( x[id(i - 1, j, k)] + x[id(i + 1, j, k)] + 
                                                 x[id(i, j - 1, k)] + x[id(i, j + 1, k)] + 
                                                 x[id(i, j, k - 1)] + x[id(i, j, k + 1)] ) 
                                           + x0[id(i, j, k)] ) / c;
                    }
                }
            }
            this.setBoundry(b, x);  
        }
    }

    void setBoundry(int b, float[] field)
    {
        int maxX = this.size.x + 1;  
        int maxY = this.size.y + 1;  
        int maxZ = this.size.z + 1;  

        for (int yIdx = 0; yIdx <= maxY; yIdx++)
        {
            for (int zIdx = 0; zIdx <= maxZ; zIdx++)
            {
                field[id(0, yIdx, zIdx)] = field[id(1, yIdx, zIdx)];           
                field[id(maxX, yIdx, zIdx)] = field[id(maxX - 1, yIdx, zIdx)]; 
            }
        }

        for (int xIdx = 0; xIdx <= maxX; xIdx++)
        {
            for (int zIdx = 0; zIdx <= maxZ; zIdx++)
            {
                field[id(xIdx, 0, zIdx)] = field[id(xIdx, 1, zIdx)];            
                field[id(xIdx, maxY, zIdx)] = field[id(xIdx, maxY - 1, zIdx)];  
            }
        }

        for (int xIdx = 0; xIdx <= maxX; xIdx++)
        {
            for (int yIdx = 0; yIdx <= maxY; yIdx++)
            {
                field[id(xIdx, yIdx, 0)] = field[id(xIdx, yIdx, 1)];           
                field[id(xIdx, yIdx, maxZ)] = field[id(xIdx, yIdx, maxZ - 1)];
            }
        }
    }


    /* x, y, z - arrays which are storing velocity components u, v, w
     * p - temporary array for computation
     * div - temporary array to hold divergence 
     **/
    public void project(float[] x, float[] y, float[] z, float[] p, float[] div)
    {
        for (int i = 1; i <= this.size.x; i++)
        {
            for (int j = 1; j <= this.size.y; j++)
            {
                for (int k = 1; k <= this.size.z; k++)
                {
                    div[id(i, j, k)] = ( x[id(i + 1, j, k)] - x[id(i - 1, j, k)] + 
                                         y[id(i, j + 1, k)] - y[id(i, j - 1, k)] + 
                                         z[id(i, j, k + 1)] - z[id(i, j, k - 1)]) / -(size.x + size.y + size.z);
                    p[id(i, j, k)] = 0;
                }
            }
        }

        this.setBoundry(0, div);
        this.setBoundry(0, p);

        this.linearSolver(0, p, div, 1, 6);

        for (int i = 1; i <= this.size.x; i++)
        {
            for (int j = 1; j <= this.size.y; j++)
            {
                for (int k = 1; k <= this.size.z; k++)
                {
                    x[id(i, j, k)] -= 0.5f * this.size.x * (p[id(i + 1, j, k)] - p[id(i - 1, j, k)]);
                    y[id(i, j, k)] -= 0.5f * this.size.y * (p[id(i, j + 1, k)] - p[id(i, j - 1, k)]);
                    z[id(i, j, k)] -= 0.5f * this.size.z * (p[id(i, j, k + 1)] - p[id(i, j, k - 1)]);
                }
            }
        }

        this.setBoundry(1, x);
        this.setBoundry(2, y);
        this.setBoundry(3, z);
    }


    /*
     * b - flag for boundries
     * d - array to store advected field
     * d0 - array to advect
     * du, dv - components of velocity
     **/

    public void advect(int b, float[] d, float[] d0, float[] du, float[] dv, float[] dw)
    {
        int i0, j0, k0, i1, j1, k1;
        float x, y, z, r0, s0, t0, r1, s1, t1, dt0;

        dt0 = this.dt * this.size.x;

        for (int i = 1; i <= this.size.x; i++)
        {
            for (int j = 1; j <= this.size.y; j++)
            {
                for (int k = 1; k <= this.size.z; k++)
                {
                    // go backwards through velocity field
                    x = i - dt0 * du[id(i, j, k)];
                    y = j - dt0 * dv[id(i, j, k)];
                    z = k - dt0 * dw[id(i, j, k)];

                    // interpolate results
                    if (x > this.size.x + 0.5)
                    {
                        x = this.size.x + 0.5f;
                    }
                    if (x < 0.5)
                    {
                        x = 0.5f;
                    }
                    i0 = (int)x;  
                    i1 = i0 + 1;

                    if (y > this.size.y + 0.5)
                    {
                        y = this.size.y + 0.5f;
                    }
                    if (y < 0.5)
                    {
                        y = 0.5f;
                    }
                    j0 = (int)y;  
                    j1 = j0 + 1;

                    if (z > this.size.z + 0.5)
                    {
                        z = this.size.z + 0.5f;
                    }
                    if (z < 0.5)
                    {
                        z = 0.5f;
                    }
                    k0 = (int)z;  
                    k1 = k0 + 1;

                    r1 = x - i0;  
                    r0 = 1 - r1;
                    s1 = y - j0;
                    s0 = 1 - s1;
                    t1 = z - k0;
                    t0 = 1 - t1;

                    d[id(i, j, k)] = r0 * (s0 * (t0 * d0[id(i0, j0, k0)] + t1 * d0[id(i0, j0, k1)])
                                            + s1 * (t0 * d0[id(i0, j1, k0)] + t1 * d0[id(i0, j1, k1)]))
                                   + r1 * (s0 * (t0 * d0[id(i1, j0, k0)] + t1 * d0[id(i1, j0, k1)])
                                            + s1 * (t0 * d0[id(i1, j1, k0)] + t1 * d0[id(i1, j1, k1)]));
                }
            }
        }
        this.setBoundry(b, d); 
    }

    public void velocitySolver()
    {
        // add velocity that was input by mouse）
        this.addSource(this.u, this.uOld);  
        this.addSource(this.v, this.vOld);
        this.addSource(this.w, this.wOld);

        // add in vorticity confinement force
        this.vorticityConfinement(this.uOld, this.vOld, this.wOld);
        this.addSource(this.u, this.uOld);
        this.addSource(this.v, this.vOld);
        this.addSource(this.w, this.wOld);

        // add in buoyancy force
        this.buoyancy(this.vOld);
        this.addSource(this.v, this.vOld);

        // swapping arrays for economical mem use
        // and calculating diffusion in velocity.
        this.swapU();
        this.diffuse(1, this.u, this.uOld, this.visc);

        this.swapV();
        this.diffuse(2, this.v, this.vOld, this.visc);

        this.swapW();
        this.diffuse(3, this.w, this.wOld, this.visc);

        this.project(this.u, this.v, this.w, this.temp, this.vOld);

        this.swapU();
        this.swapV();
        this.swapW();
        this.advect(1, this.u, this.uOld, this.uOld, this.vOld, this.wOld);
        this.advect(2, this.v, this.vOld, this.uOld, this.vOld, this.wOld);
        this.advect(3, this.w, this.wOld, this.uOld, this.vOld, this.wOld);

        // make an incompressible field
        this.project(this.u, this.v, this.w, this.temp, this.vOld);

        // clear all input velocities for next frame
        for (int x = 0; x < this.bSize.x; x++)
        {
            for (int y = 0; y < this.bSize.y; y++)
            {
                for (int z = 0; z < this.bSize.z; z++)
                {
                    this.uOld[id(x, y, z)] = 0;
                    this.vOld[id(x, y, z)] = 0;
                    this.wOld[id(x, y, z)] = 0;
                }
            }
        }
    }
    
    public void densitySolver()
    {
        // fire source & ventilation
        /*if (this.simulationTime < 50)
        {
            this.d[id(this.bSize * 2 / 3, this.bSize - 2, this.bSize / 2)] = 100;
        }
        else
        {
            for (int i = (int)(0.15 * this.size); i < 0.55 * this.size; ++i)
            {
                for (int j = (int)(0.35 * this.size); j < 0.65 * this.size; ++j)
                {
                    this.d[id(0, i, j)] = 0;
                    this.d[id(1, i, j)] = 0;
                }
            }
        }*/
        
        this.addSource(this.d, this.dOld);

        this.swapD();

        this.diffuse(0, this.d, this.dOld, this.diff);

        this.swapD();

        this.advect(0, this.d, this.dOld, this.u, this.v, this.w);

        for (int x = 0; x < this.bSize.x; x++)
        {
            for (int y = 0; y < this.bSize.y; y++)
            {
                for (int z = 0; z < this.bSize.z; z++)
                {
                    this.dOld[id(x, y, z)] = 0;
                }
            }
        }
    }
}
