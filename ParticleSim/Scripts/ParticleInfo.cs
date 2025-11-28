using UnityEngine;

[System.Serializable]
public struct ParticleInfo
{
    public Vector3 position;            
    public Vector3 initialVelocity;     
    public int generationTimeStamp;      
    public Vector3 initialPosition;     
    public int generationCode;          
    public int particleGenerationIndex; 
}
