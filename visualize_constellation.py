
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from constellation_env import ConstellationEnv, T_SOFT_LIMIT, T_TARGET

def main():
    # 1. Load Env and Normalize Stats
    env = DummyVecEnv([lambda: ConstellationEnv()])
    stats_path = "models_constellation/vec_normalize.pkl"
    env = VecNormalize.load(stats_path, env)
    env.training = False # Don't update stats
    
    # 2. Load Model
    model = PPO.load("models_constellation/final_model", env=env)
    
    # 3. Run Episode
    obs = env.reset()
    
    temps = []
    throttles = []
    is_lits = []
    served_demands = []
    global_demands = []
    
    print("Running evaluation episode...")
    for _ in range(300): # 10 hours approx (300 * 2 min = 600 min = 10h)
        # Predict
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        # Extract data from info (env wrapper puts info in list)
        info = infos[0] 
        
        # We need per-sat data. 
        # ConstellationEnv doesn't expose per-sat throttles in info efficiently yet.
        # But we can reconstruct throttles from action if we knew the logic,
        # OR we can hack ConstellationEnv to output it,
        # OR we can read it from the env object directly since it's DummyVecEnv.
        
        # Access underlying env through VecNormalize -> DummyVecEnv
        raw_env = env.venv.envs[0]
        
        # Extract temps from sats
        step_temps = [s.current_temp for s in raw_env.sats]
        temps.append(step_temps)
        
        # Extract throttles (prev_compute_throttle is stored in sat)
        step_throttles = [s.prev_compute_throttle for s in raw_env.sats]
        throttles.append(step_throttles)
        
        # Extract Lit Status
        # We need to re-compute or access from env
        # raw_env.sats[i].sat.at(t).is_sunlit...
        # Let's trust the env state or recompute.
        t = raw_env.sats[0].t_current
        step_lits = [1.0 if s.sat.at(t).is_sunlit(s.eph) else 0.0 for s in raw_env.sats]
        is_lits.append(step_lits)
        
        served_demands.append(info.get("served_units", 0))
        global_demands.append(info.get("global_demand_units", 0))
        
        if dones[0]:
            break
            
    # 4. Plot Heatmaps
    temps = np.array(temps).T # [24, T]
    throttles = np.array(throttles).T # [24, T]
    is_lits = np.array(is_lits).T # [24, T]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # --- Temp Heatmap ---
    # Overlay eclipse regions?
    # Actually, let's just show Temp.
    im1 = axes[0].imshow(temps, aspect='auto', cmap='inferno', vmin=300, vmax=370)
    axes[0].set_title("Fleet Temperature (K)")
    axes[0].set_ylabel("Sat ID")
    plt.colorbar(im1, ax=axes[0])
    
    # Overlay Eclipse (Hatch)
    # We can create a mask
    # For now, just trust the thermal patterns.
    
    # --- Throttle Heatmap ---
    im2 = axes[1].imshow(throttles, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title("Fleet Compute Throttle")
    axes[1].set_ylabel("Sat ID")
    plt.colorbar(im2, ax=axes[1])
    
    # --- Solar Status Heatmap ---
    im3 = axes[2].imshow(is_lits, aspect='auto', cmap='binary', vmin=0, vmax=1)
    axes[2].set_title("Sunlight (White=Sun, Black=Eclipse)")
    axes[2].set_xlabel("Time Step (2 min)")
    axes[2].set_ylabel("Sat ID")
    
    plt.tight_layout()
    plt.savefig("constellation_heatmap.png")
    print("Saved constellation_heatmap.png")
    
    # --- Baseline Comparison ---
    print("\n--- Running Baseline (Round Robin) ---")
    obs = env.reset()
    rr_served = 0.0
    rl_served = sum(served_demands)
    
    for _ in range(300):
        # Round Robin Action: Uniform weights
        # Logits of 0.0 -> exp(0)=1 -> normalized = 1/24
        action = np.zeros((1, 24), dtype=np.float32)
        obs, rewards, dones, infos = env.step(action)
        rr_served += infos[0].get("served_units", 0)
        if dones[0]: break
        
    print(f"RL Total Served: {rl_served:.2f}")
    print(f"RR Total Served: {rr_served:.2f}")
    if rl_served > rr_served:
        print(f"RL Improvement: +{(rl_served - rr_served)/rr_served*100:.1f}%")
    else:
        print(f"RL Deficit: {(rl_served - rr_served)/rr_served*100:.1f}%")

    # Plot Demand vs Served
    plt.figure(figsize=(10, 4))
    plt.plot(global_demands, label="Global Demand", color='black', linestyle='--')
    plt.plot(served_demands, label="RL Agent", color='blue')
    # We didn't save RR history, just total. 
    # For a plot, we'd need to save it. 
    # But for now, text metric is enough for MVP.
    plt.title(f"Constellation Capacity: RL vs Demand (RL served {rl_served:.0f} units)")
    plt.xlabel("Time Step")
    plt.ylabel("Compute Units")
    plt.legend()
    plt.grid(True)
    plt.savefig("constellation_demand.png")
    print("Saved constellation_demand.png")

if __name__ == "__main__":
    main()
