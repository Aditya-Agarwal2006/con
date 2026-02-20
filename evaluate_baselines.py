import gymnasium as gym
import numpy as np
import pandas as pd
from constellation_env import ConstellationEnv
from sat_env import T_TARGET, T_SOFT_LIMIT, T_HARD_LIMIT

def evaluate_policy(env, policy_fn, episodes=5, max_steps=1000, name="Baseline"):
    print(f"--- Evaluating {name} ---")
    results = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        thermal_violations = 0
        total_throughput = 0.0
        total_coverage = 0.0
        total_steps = 0
        done = False
        
        while not done and total_steps < max_steps:
            action = policy_fn(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            
            ep_reward += reward
            total_throughput += info.get("total_throughput", 0.0)
            total_coverage += info.get("coverage_fraction", 0.0)
            
            # Check thermal violations (temp > soft limit)
            # Obs layout: 1 global demand, then blocks of 15 per sat. Temp is index 0 of sat block.
            # Convert normal temp back to real temp: T = (norm * 200) + 200
            for i in range(env.n_sats):
                t_norm = obs[1 + i*15 + 0]
                t_real = (t_norm * 200) + 200
                if t_real > T_SOFT_LIMIT:
                    thermal_violations += 1
                    
            total_steps += 1
            done = terminated or truncated
            
        metrics = {
            "Policy": name,
            "Episode": ep,
            "Total Reward": ep_reward,
            "Avg Throughput/Step": total_throughput / (total_steps + 1e-6),
            "Avg Coverage/Step": total_coverage / (total_steps + 1e-6),
            "Thermal Violations": thermal_violations,
            "Steps Completed": total_steps
        }
        results.append(metrics)
        print(f"  Ep {ep}: R={ep_reward:.1f}, Steps={total_steps}, Violations={thermal_violations}, AvgThrp={metrics['Avg Throughput/Step']:.1f}")
        
    return pd.DataFrame(results)

# --- BASELINE POLICIES ---

def random_policy(obs, env):
    # Action space: 120D continuous [-0.1..0.1, ... 0..1, 0..pi]
    return env.action_space.sample()

def walker_standard_policy(obs, env):
    # Static Walker: No thrust, 100% compute (throttle=1.0), face sun (theta=0)
    # Action format per sat: [thrust_r, thrust_t, thrust_n, throttle, theta]
    action = np.zeros((env.n_sats, 5), dtype=np.float32)
    action[:, 3] = 1.0 # 100% throttle
    return action.flatten()

def rule_based_thermal_policy(obs, env):
    # Heuristic: 
    # If temp > 340K -> compute to 50%
    # If temp > 350K -> compute to 0%, turn edge-on to sun (theta = pi/2)
    action = np.zeros((env.n_sats, 5), dtype=np.float32)
    
    for i in range(env.n_sats):
        t_norm = obs[1 + i*15 + 0]
        t_real = (t_norm * 200) + 200
        
        throttle = 1.0
        theta = 0.0
        
        if t_real > 350.0:
            throttle = 0.0
            theta = np.pi / 2.0
        elif t_real > 340.0:
            throttle = 0.5
            
        action[i, 3] = throttle
        action[i, 4] = theta
        
    return action.flatten()

def main():
    env = ConstellationEnv()
    
    print("Pre-computing Baseline Metrics (Episodes=3 for speed)...")
    
    df_random = evaluate_policy(env, random_policy, episodes=3, max_steps=500, name="Random")
    df_walker = evaluate_policy(env, walker_standard_policy, episodes=3, max_steps=500, name="Walker Standard")
    df_rule = evaluate_policy(env, rule_based_thermal_policy, episodes=3, max_steps=500, name="Rule-Based")
    
    df_all = pd.concat([df_random, df_walker, df_rule], ignore_index=True)
    df_all.to_csv("baselines_evaluation.csv", index=False)
    print("\nSaved evaluation results to baselines_evaluation.csv")

if __name__ == "__main__":
    main()
