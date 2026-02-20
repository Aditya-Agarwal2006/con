import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

# Import your custom environment
# Ensure sat_env.py is in the same folder
from sat_env import SatEnv 

# --- CONFIGURATION ---
TIMESTEPS = 1000000
LOG_DIR = "./logs"
MODEL_DIR = "./models"

# Create directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("--- STARTING TRAINING SESSION ---")
    
    # 1. Vectorize the Environment
    # This runs 4 environments in parallel to speed up training
    # (Your CPU has plenty of cores for this)
    env = make_vec_env(lambda: SatEnv(), n_envs=8)
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    # 2. Initialize the Agent (The Brain)
    # MlpPolicy = Multi-Layer Perceptron (Standard Neural Network)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=LOG_DIR,
        learning_rate=1e-4,
        batch_size=1024,
        n_steps=1024,
        n_epochs=10,
        ent_coef=0.01,
        gamma=0.99
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // 8,  # Account for 8 parallel envs
        save_path=MODEL_DIR,
        name_prefix="ppo_sat_ckpt",
    )

    # 3. Train!
    print(f"Training for {TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True, callback=checkpoint_callback)

    # 4. Save the Result
    model_name = "ppo_sat_v2"
    model.save(f"{MODEL_DIR}/{model_name}")
    env.save(f"{MODEL_DIR}/{model_name}_vecnormalize.pkl")
    print(f"Training Complete! Model saved to {MODEL_DIR}/{model_name}.zip")

if __name__ == "__main__":
    main()