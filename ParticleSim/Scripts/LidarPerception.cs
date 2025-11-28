using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Linq; 
using Random = UnityEngine.Random;
using ComputeBufferParticleInfoStruct = ParticleInfo;

public class LidarPerception : MonoBehaviour
{
    [Header("SmokeParticlesReference")]
    [SerializeField]
    private SmokeParticlesFromManager m_SmokeParticlesModel = null;


    [Header("Lidar Parameters")]
    [SerializeField]
    private TextAsset m_InclinationData = null;   
    [SerializeField]
    private int m_NumberOfCircularSegments = 2650; 
    [SerializeField]
    private float m_MaxRange = 75.0f; 
    [SerializeField]
    [Range(0.0f, 1.0f)]
    private float m_GeneralDropOffRate = 0.17f;  


    [Header("Render Entities")]
    [SerializeField]
    private Material m_RenderMaterial = null;  
    [SerializeField]
    private Mesh m_PointCloudEntityMesh = null;   


    [Header("Compute Shaders")]
    [SerializeField]
    private ComputeShader m_RaycastComputeShader = null;  


    [Header("Export Config")]
    [SerializeField]
    private int m_StoppingFrameCount = 1000; 
    [SerializeField]
    private bool m_IsExportingPointCloud = true;   

    // Lidar simulation data
    private float m_azimuthIncrementAngle;  
    private List<float> m_inclinationAngles = new List<float>(); 
    private const int particleInfoSize = 50000 * 100; 
    private int m_rayCount = 0;        

    private ComputeBuffer m_particleInfoBuffer = null;  
    private ComputeBuffer m_raycastInfoBuffer = null;   
    private ComputeBuffer m_raycastResultInfoBuffer = null; 

    ComputeBufferParticleInfoStruct[] m_particleInfoDataArray = null;           
    ComputeBufferRaycastInfoStruct[] m_raycastInfoDataArray = null;              
    ComputeBufferRaycastResultInfoStruct[] m_raycastResultInfoDataArray = null;    


    // Rendering point cloud
    //private List<Vector3> m_pointCloudPositions = new List<Vector3>();  
    private ComputeBuffer argsBuffer;                                   
    private uint[] args = new uint[5] { 0, 0, 0, 0, 0 };                


    // Exporting point cloud
    private int m_currentPointCloudIndex = 0;  


    // add 
    private float m_LidarScanTimeStep = 0.1f;   
    //private float m_LidarScanTimeStep = 0.2f;
    private float m_GroundPlaneHeight = -1.0f;
    //private float m_GroundPlaneHeight = -0.0f;
    string m_OutputPCDPath = "/home/project/zsj/Unity3D/SmokeCFD/SmokeParticleCFD/Assets/Scripts/OutputPCD/";
    
    // Set the scanning time
    public SmokeManager smokeManager;        
    public float startScanTimeThreshold = 10f; 
    private bool scanningEnabled = false;


    private void Awake()
    {
        m_azimuthIncrementAngle = (360.0f / m_NumberOfCircularSegments);  
        InitializeInclinationAngles();  
        InitializeComputeBuffer();      
    }

    private void OnDestroy()
    {
        if (m_particleInfoBuffer != null)
        {
            m_particleInfoBuffer.Release();
            m_particleInfoBuffer = null;   
        }

        if (m_raycastInfoBuffer != null)
        {
            m_raycastInfoBuffer.Release();
            m_raycastInfoBuffer = null;
        }

        if (m_raycastResultInfoBuffer != null)
        {
            m_raycastResultInfoBuffer.Release();
            m_raycastResultInfoBuffer = null;
        }

        if (argsBuffer != null)
        {
            argsBuffer.Release();
            argsBuffer = null;
        }
    }

    
    private void InitializeInclinationAngles()
    {
        string allText = m_InclinationData.text; 
        string[] lines = allText.Split('\n');     
                                     
        // Skip idx 0 as it is the header line in the .csv file
        for (int idx = 0; idx < lines.Length; ++idx) 
        {
            //string data = lines[idx];
            string data = lines[idx].Trim();
            float inclination = -1.0f;
            if (float.TryParse(data, out inclination) != true)  
            {
                Debug.LogError(string.Format("Error when parsing inclination file for string : {0}", data));
            }
            else
            {
                float inclinationInDeg = Mathf.Rad2Deg * inclination; 
                m_inclinationAngles.Add(inclinationInDeg); 
            }
        }
    }


    private void InitializeComputeBuffer()
    {
        argsBuffer = new ComputeBuffer(1, args.Length * sizeof(uint), ComputeBufferType.IndirectArguments);

        // Compute how many rays we need
        m_rayCount = 0;
        List<Vector3> directions = new List<Vector3>();
        for (int incr = 0; incr < m_NumberOfCircularSegments; incr++) 
        {
            for (int layer = 0; layer < m_inclinationAngles.Count; layer++)  
            {
                int index = layer + incr * m_inclinationAngles.Count;
                float angle = m_inclinationAngles[layer];
                float azimuth = incr * m_azimuthIncrementAngle;

                // Quaternion.Euler(x, y, z) 
                Vector3 dir = transform.rotation * Quaternion.Euler(-angle, azimuth, 0) * transform.forward;
                //Vector3 dir = transform.rotation * Quaternion.Euler(angle, azimuth, 0) * transform.forward;

                // Skip all fowward directions
                //if (Vector3.Angle(transform.forward, dir) < 120.0f)
                //    continue;
                // if(angle < -13.0f && Mathf.Abs(azimuth - 180) < 20.0f) continue;

                ++m_rayCount;
                directions.Add(dir);

                //float verticalAngle = Vector3.SignedAngle(Vector3.forward, dir, Vector3.right);
            }
        }


        m_particleInfoBuffer = new ComputeBuffer(particleInfoSize, 48); 
        
        if (m_rayCount > 0)
        {
            m_raycastInfoBuffer = new ComputeBuffer(m_rayCount, 24); 
            m_raycastResultInfoBuffer = new ComputeBuffer(m_rayCount, 52); 
        }

        // Initialize the raycastInfo compute buffer data
        m_particleInfoDataArray = new ComputeBufferParticleInfoStruct[particleInfoSize];
        if (m_rayCount > 0)
        {
            m_raycastInfoDataArray = new ComputeBufferRaycastInfoStruct[m_rayCount];
            m_raycastResultInfoDataArray = new ComputeBufferRaycastResultInfoStruct[m_rayCount];
        }

        for (int rayIndex = 0; rayIndex < m_rayCount; ++rayIndex)
        {
            ComputeBufferRaycastInfoStruct raycastInfo = new ComputeBufferRaycastInfoStruct
            {
                origin = transform.position,      
                direction = directions[rayIndex]  
            };
            m_raycastInfoDataArray[rayIndex] = raycastInfo;

            ComputeBufferRaycastResultInfoStruct raycastResultInfo = new ComputeBufferRaycastResultInfoStruct
            {
                distance = 0.0f,           
                hitPosition = Vector3.zero  
            };
            m_raycastResultInfoDataArray[rayIndex] = raycastResultInfo;
        }
        
        m_raycastInfoBuffer.SetData(m_raycastInfoDataArray);
        m_raycastResultInfoBuffer.SetData(m_raycastResultInfoDataArray);

        
        // Initialize the particleInfo compute buffer data
        for (int particleIndex = 0; particleIndex < particleInfoSize; ++particleIndex)
        {
            ComputeBufferParticleInfoStruct particleInfo = new ComputeBufferParticleInfoStruct
            {
                position = Vector3.zero,           
                initialVelocity = Vector3.zero,   
                generationTimeStamp = -1,          
                initialPosition = Vector3.zero,
                generationCode = 0,                
                particleGenerationIndex = 0         
            };

            m_particleInfoDataArray[particleIndex] = particleInfo;
        }
        m_particleInfoBuffer.SetData(m_particleInfoDataArray);
    }


    private float m_UpdateLidarTimer = 0.0f;  
    //private void FixedUpdate()  
    private void Update() 
    {
        if (!scanningEnabled && smokeManager.simulationTime >= startScanTimeThreshold)
        {
            scanningEnabled = true;
        }

        if (!scanningEnabled) return;
    
        if (m_UpdateLidarTimer >= m_LidarScanTimeStep)
        {
            IRUpdate();  
            m_UpdateLidarTimer = 0.0f; 
        }
        m_UpdateLidarTimer += Time.fixedDeltaTime;  
    }

    private void IRUpdate()
    {
        UpdateLidarDetection();
       
        m_RenderMaterial.SetBuffer("raycastResultInfoBuffer", m_raycastResultInfoBuffer);
        args[0] = (uint)m_PointCloudEntityMesh.GetIndexCount(0);    
        args[1] = (uint)m_raycastResultInfoDataArray.Length;       
        args[2] = (uint)m_PointCloudEntityMesh.GetIndexStart(0);   
        args[3] = (uint)m_PointCloudEntityMesh.GetBaseVertex(0);    
        argsBuffer.SetData(args); 
        
        if (m_IsExportingPointCloud == true)
        {）
            ExportPointCloud(
                string.Format("result{0}.pcd", m_currentPointCloudIndex.ToString("D4")),
                m_raycastResultInfoDataArray                   
                );     
        }

        if (m_currentPointCloudIndex >= m_StoppingFrameCount)
        {
            Application.Quit();
        }

        ++m_currentPointCloudIndex;
    }

    // This loop will update the distance array
    private void UpdateLidarDetection()
    {
        // Update collider info buffers, particles and triangle mesh
        List<ParticleInfo> points = new List<ParticleInfo>();
        points.AddRange(m_SmokeParticlesModel.AllParticleInfo);  
        //Debug.Log(points.Count); 
        //for (int particleIndex = 0; particleIndex < points.Count; ++particleIndex)
        //{
            //Debug.Log(points[particleIndex].position);
        //}
        UpdateParticleBuffer(points); 

        // Initialize the particleInfo compute buffer data
        m_RaycastComputeShader.SetFloat("maxRange", m_MaxRange + 0.005f); 
        m_RaycastComputeShader.SetFloat("time", Time.time);  
        m_RaycastComputeShader.SetFloat("generalDropOffRate", m_GeneralDropOffRate);  
        m_RaycastComputeShader.SetFloat("vehicleVelocity", m_SmokeParticlesModel.VehicleVelocity);  

        // This should match the rotation frequency of the simulating lidar, which in waymo is 10 hz,
        // which makes the time spent on a single lidar scan 0.1 seconds.
        m_RaycastComputeShader.SetFloat("lidarScanTime", m_LidarScanTimeStep); 
        m_RaycastComputeShader.SetInt("particleCount", points.Count);   

        //////////////////// UPDATE : for mani check, remove if using for recon or synthesis
        m_RaycastComputeShader.SetFloat("particleCollideRadius", 0.035f);
        //m_RaycastComputeShader.SetFloat("particleCollideRadius", m_SmokeParticlesModel.ParticleSize / 30.0f * 0.035f);
        ////////////////////

        // Set the random offsets needed for the gaussrand in compute shader
        m_RaycastComputeShader.SetVector("offsets", new Vector4(
            Random.Range(-1.0f, 1.0f),  
            Random.Range(-1.0f, 1.0f),  
            Random.Range(-1.0f, 1.0f), 
            Random.Range(-1.0f, 1.0f)   
        ));
        m_RaycastComputeShader.SetFloat("groundHeight", m_GroundPlaneHeight);  

        // Call compute shader, assign data to m_distances
        int computeShaderEntry = m_RaycastComputeShader.FindKernel("CSMain");  
        m_RaycastComputeShader.SetBuffer(computeShaderEntry, "particleInfoBuffer", m_particleInfoBuffer); 
        m_RaycastComputeShader.SetBuffer(computeShaderEntry, "raycastInfoBuffer", m_raycastInfoBuffer);   
        m_RaycastComputeShader.SetBuffer(computeShaderEntry, "raycastResultInfoBuffer", m_raycastResultInfoBuffer);  

        m_RaycastComputeShader.Dispatch(computeShaderEntry, m_rayCount / 64, 1, 1); 

        // Update computed data 
        m_raycastResultInfoBuffer.GetData(m_raycastResultInfoDataArray);
    }

    private void UpdateParticleBuffer(List<ParticleInfo> points)
    {
        if (points.Count >= m_particleInfoDataArray.Length)
        {
            Debug.LogError(
                "UpdateParticleBuffer : points.Count >= m_particleInfoDataArray.Count : "
                + "points.Count = " + points.Count.ToString()
                + "m_particleInfoDataArray.Count = " + m_particleInfoDataArray.Length.ToString()
                );
        }

        for (int particleIndex = 0; particleIndex < points.Count; ++particleIndex)
        {
            m_particleInfoDataArray[particleIndex] = points[particleIndex];
        }
        m_particleInfoBuffer.SetData(m_particleInfoDataArray);
    }

    public void ExportPointCloud(
            string fileName,
            ComputeBufferRaycastResultInfoStruct[] raycastResultInfoDataArray)
    {
        StreamWriter writer = new StreamWriter(m_OutputPCDPath + fileName);

        // Count the number of valid points
        int validPointCount = 0;
        float minY = 1000, maxY = -1000;
        foreach (var entry in raycastResultInfoDataArray)
        {
            ++validPointCount;
            
            float y = entry.hitPosition.y;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }
        
        writer.WriteLine("VERSION 0.7");
        writer.WriteLine("FIELDS x y z");
        writer.WriteLine("SIZE 4 4 4");
        writer.WriteLine("TYPE F F F");
        writer.WriteLine("COUNT 1 1 1");

        writer.WriteLine("WIDTH " + validPointCount);
        writer.WriteLine("HEIGHT 1");
        writer.WriteLine("VIEWPOINT 0 0 0 1 0 0 0");
        writer.WriteLine("POINTS " + validPointCount);
        writer.WriteLine("DATA ascii");

        foreach (var entry in raycastResultInfoDataArray)
        {
            Vector3 transformed = TransformPointToWaymoSpace(entry.hitPosition);
            writer.WriteLine(string.Format("{0} {1} {2}", transformed.x, transformed.y, transformed.z));
        }

        writer.Close();
    }

    public Vector3 TransformPointToWaymoSpace(Vector3 point)
    {
        return new Vector3(
        point.z,
        -1.0f * point.x,   
        point.y - m_GroundPlaneHeight);
    }


    // size = size of 2 Vector3 = 4*3 * 2 = 24
    struct ComputeBufferRaycastInfoStruct
    {
        public Vector3 origin;
        public Vector3 direction;
    }


    // size = size of 3 Vector3 + size of 2 float + 2 int = 4*3*3 + 4*2 + 4*2 = 52
    public struct ComputeBufferRaycastResultInfoStruct
    {
        public float distance;
        public Vector3 hitPosition;
        public int hitParticleGenerationTimeStamp;
        public Vector3 hitParticleInitialVelocity;
        public Vector3 hitParticleInitialPosition;
        public int hitParticleGenerationCode;
        public int hitParticleGenerationIndex;
    }

}



