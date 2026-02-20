import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from constellation_env import ConstellationEnv
from sat_env import T_TARGET, T_SOFT_LIMIT, T_HARD_LIMIT
from train_constellation import ConstellationFeatureExtractor

def evaluate_rl_model(model_path, vec_env_path, model_name="Naive_RL", episodes=3, max_steps=500):
    print(f"--- Evaluating RL Model: {model_name} ---")
    
    # Needs to match the environment kwargs used during training evaluation
    # Since this is to test the model, we use the standard ConstellationEnv (no naive_thermal flag! 
    # the thermal-naive agent must face the standard thermal laws to see its errors)
    env = DummyVecEnv([lambda: ConstellationEnv(naive_thermal=False)])
    env = VecNormalize.load(vec_env_path, env)
    env.training = False
    env.norm_reward = False
    
    # Policy kwargs need to be identical to what trained the model
    # Note: custom extractor should be loaded automatically by PPO.load if class is in scope
    model = PPO.load(model_path, env=env)
    
    results = []
    
    for ep in range(episodes):
        obs = env.reset()
        ep_reward = 0.0
        thermal_violations = 0
        total_throughput = 0.0
        total_coverage = 0.0
        total_steps = 0
        done = False
        
        while not done and total_steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            
            info = infos[0] # Single env unwrapped
            ep_reward += reward[0]
            total_throughput += info.get("total_throughput", 0.0)
            total_coverage += info.get("coverage_fraction", 0.0)
            
            raw_env = env.venv.envs[0]
            for sat in raw_env.sats:
                if sat.current_temp > T_SOFT_LIMIT:
                    thermal_violations += 1
                    
            total_steps += 1
            done = dones[0]
            
        metrics = {
            "Policy": model_name,
            "Episode": ep,
            "Total Reward": ep_reward,
            "Avg Throughput/Step": total_throughput / (total_steps + 1e-6),
            "Avg Coverage/Step": total_coverage / (total_steps + 1e-6),
            "Thermal Violations": thermal_violations,
            "Steps Completed": total_steps
        }
        results.append(metrics)
        print(f"  Ep {ep}: R={ep_reward:.1f}, Steps={total_steps}, Violations={thermal_violations}, AvgThrp={metrics['Avg Throughput/Step']:.1f}")
        
    df = pd.DataFrame(results)
    
    # Append to existing baseline CSV
    try:
        df_old = pd.read_csv("baselines_evaluation.csv")
        df_all = pd.concat([df_old, df], ignore_index=True)
        df_all.to_csv("baselines_evaluation.csv", index=False)
        print("Updated baselines_evaluation.csv")
    except Exception as e:
        df.to_csv("baselines_evaluation.csv", index=False)
        print("Saved NEW baselines_evaluation.csv")
        
    return df

if __name__ == "__main__":
    evaluate_rl_model(
        model_path="models_naive/final_model.zip",
        vec_env_path="models_naive/vec_normalize.pkl",
        model_name="Thermal-Naive RL"
    )
