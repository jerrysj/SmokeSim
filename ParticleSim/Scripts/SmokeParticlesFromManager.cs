using System.Collections.Generic;
using UnityEngine;

public class SmokeParticlesFromManager : MonoBehaviour
{
    [SerializeField] private ParticleManager particleManager;
    private float m_Velocity = 10.0f; 
    
    public ParticleSystem particleSystemRef;

    private ParticleSystem.Particle[] particlesBuffer;
    private List<ParticleInfo> m_allParticleInfo = new List<ParticleInfo>();

    public List<ParticleInfo> AllParticleInfo
    {
        get
        {
            UpdateAllParticleInfo();
            return m_allParticleInfo;
        }
    }

    private void UpdateAllParticleInfo()
    {
        if (particleSystemRef == null)
        {
            Debug.LogWarning("[SmokeParticlesFromManager] 未绑定粒子系统引用。");
            return;
        }

        int aliveCount = particleSystemRef.particleCount;

        if (particlesBuffer == null || particlesBuffer.Length < aliveCount)
            particlesBuffer = new ParticleSystem.Particle[aliveCount * 2];

        int count = particleSystemRef.GetParticles(particlesBuffer);
        m_allParticleInfo.Clear();
        for (int i = 0; i < count; i++)
        {
            var p = particlesBuffer[i];
            m_allParticleInfo.Add(new ParticleInfo
            {
                position = p.position,
                initialVelocity = p.velocity,
                generationTimeStamp = Time.frameCount,
                initialPosition = p.position,
                generationCode = 0,
                particleGenerationIndex = i
            });
        }
    }

    public float ParticleSize
    {
        get
        {
            if (particleManager != null)
                return particleManager.ParticleSize;
            else
                return 0.04f;
        }
    }

    public float VehicleVelocity 
    {
        get { return m_Velocity; }
        set { m_Velocity = value; }
    }
}
