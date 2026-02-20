# Reinforcement Learning for Space-Based GPU Datacenters

## Concept
Can we run AI datacenters in orbit? This simulates thermal management for GPU compute in space.

## Physics Engine
- Stefan-Boltzmann radiative cooling
- NVIDIA H100 thermal model (358K throttling)
- 14,000+ debris objects (Kessler syndrome constraint)
- Orbital mechanics (fuel-constrained maneuvers)

## RL Approach
**PPO agent** learns multi-objective optimization:
- Maximize compute throughput
- Stay below thermal limits
- Avoid debris collisions
- Conserve fuel

## Emergent Behavior
Agent discovered **duty cycling**: pulsing compute loads to stay cool rather than running continuously. 
*See `baseline_comparison.png` for proof of the RL agent outperforming PID and Bang-Bang controllers by utilizing orbital thermal mass.*

Currently scaling to 24-satellite constellation for distributed workload orchestration.

## Tech Stack
PyTorch, Stable Baselines3, Gymnasium, NumPy, Skyfield

## Running the Code
1. **Single Satellite Training:** `python train_ppo.py`
2. **Constellation Dispatcher Training:** `python train_constellation.py`
3. **Generate Baselines Comparison Plot:** `python compare_baselines.py`
