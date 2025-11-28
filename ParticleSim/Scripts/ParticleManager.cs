using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(ParticleSystem))]
public class ParticleManager : MonoBehaviour
{
    public FluidSolver fluidSolver;
    public SmokeManager smokeManager;

    [Header("发射设置")]
    public float densityThreshold = 0.01f;
    public int maxParticlesPerCell = 5;
    public float ParticleSize = 0.05f;
    public float baseLifetime = 1.5f;
    public float lifetimeVariation = 1.0f;

    [Header("扰动设置")]
    public float velocityJitter = 0.1f;
    public float vortexStrength = 0.2f;
    public float upwardBias = 0.5f;

    private ParticleSystem particleSys;
    private ParticleSystem.Particle[] particleBuffer;

    void Start()
    {
        particleSys = GetComponent<ParticleSystem>();
        var main = particleSys.main;
        main.startSpeed = 0;
        main.startLifetime = 5f;
        main.maxParticles = 100000;
    }

    void Update()
    {
        EmitFromFluid();
        Debug.Log($"[ParticleManager] Frame {Time.frameCount} - Alive Particles: {particleSys.particleCount}");
    }

    void EmitFromFluid()
    {
        if (fluidSolver == null || smokeManager == null || fluidSolver.size.x <= 0) return;

        for (int x = 1; x < fluidSolver.size.x - 1; x++)
        {
            for (int y = 1; y < fluidSolver.size.y - 1; y++)
            {
                for (int z = 1; z < fluidSolver.size.z - 1; z++)
                {
                    int idx = fluidSolver.id(x, y, z);
                    float density = fluidSolver.d[idx];

                    if (density < densityThreshold)
                        continue;

                    float heightFactor = Mathf.Clamp01((float)y / (fluidSolver.size.y - 1));
                    float emissionFactor = Mathf.Lerp(1.2f, 0.1f, heightFactor);
                    int emitCount = Mathf.Clamp(Mathf.RoundToInt(density * maxParticlesPerCell * emissionFactor), 0, maxParticlesPerCell);

                    Vector3 worldPos = smokeManager.gridToWorldPos(new Vector3Int(x, y, z), true);
                    Vector3 baseVel = SampleVelocity(worldPos);
                    Vector3 vortex = GenerateVortex(worldPos);

                    Vector3 upward = new Vector3(0, upwardBias * (1.0f - heightFactor), 0);
                    Vector3 finalVel = baseVel + vortex + upward;

                    float lifetime = baseLifetime + Random.Range(-lifetimeVariation, lifetimeVariation);
                    //lifetime *= Mathf.Lerp(1.2f, 0.6f, heightFactor);
                    lifetime *= Mathf.Lerp(0.1f, 10.5f, heightFactor); 

                    for (int i = 0; i < emitCount; i++)
                    {
                        ParticleSystem.EmitParams emitParams = new ParticleSystem.EmitParams
                        {
                            position = worldPos + Random.insideUnitSphere * 0.08f,
                            velocity = finalVel + Random.insideUnitSphere * velocityJitter,
                            startSize = ParticleSize,
                            startLifetime = Mathf.Max(0.2f, lifetime)
                        };

                        particleSys.Emit(emitParams, 1);
                    }
                }
            }
        }
    }

    Vector3 SampleVelocity(Vector3 pos)
    {
        Vector3 grid = smokeManager.worldToGridPos(pos);
        Vector3Int cell = new Vector3Int(
            Mathf.Clamp(Mathf.FloorToInt(grid.x), 1, fluidSolver.size.x - 2),
            Mathf.Clamp(Mathf.FloorToInt(grid.y), 1, fluidSolver.size.y - 2),
            Mathf.Clamp(Mathf.FloorToInt(grid.z), 1, fluidSolver.size.z - 2)
        );
        int idx = fluidSolver.id(cell.x, cell.y, cell.z);
        return new Vector3(fluidSolver.u[idx], fluidSolver.v[idx], fluidSolver.w[idx]);
    }

    Vector3 GenerateVortex(Vector3 pos)
    {
        float time = Time.time * 0.6f;
        float noiseX = Mathf.PerlinNoise(pos.x * 0.3f + time, pos.z * 0.3f + time);
        float noiseY = Mathf.PerlinNoise(pos.y * 0.2f + time, pos.x * 0.2f - time);
        float noiseZ = Mathf.PerlinNoise(pos.z * 0.3f - time, pos.y * 0.3f + time);

        float angleXZ = noiseX * Mathf.PI * 2;
        float angleY = (noiseY - 0.5f) * 2f; 
        Vector3 dir = new Vector3(Mathf.Cos(angleXZ), angleY, Mathf.Sin(angleXZ)).normalized;
        return dir * vortexStrength;
    }
}

