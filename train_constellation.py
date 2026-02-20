
import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from constellation_env import ConstellationEnv
from safety_filter import ThermalSafetyWrapper

class ConstellationLogCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    Logs metrics from the 'info' dict of the environment.
    """
    def __init__(self, verbose=0):
        super(ConstellationLogCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Access the infos from the vec_env
        # We assume DummyVecEnv with 1 env
        infos = self.locals["infos"]
        
        for info in infos:
            if "global_demand_units" in info:
                self.logger.record("constellation/global_demand", info["global_demand_units"])
                self.logger.record("constellation/served_demand", info["served_units"])
                self.logger.record("constellation/unserved_percent", 
                                   (info["global_demand_units"] - info["served_units"]) / (info["global_demand_units"] + 1e-6))
                
            if "avg_temp" in info:
                self.logger.record("constellation/avg_fleet_temp", info["avg_temp"])
                
            if "failed_sats" in info:
                self.logger.record("constellation/failed_sats", info["failed_sats"])
                
        return True

def main():
    # Create log dir
    log_dir = "logs_constellation/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create Environment
    # We use a lambda to ensure clean creation
    env = DummyVecEnv([lambda: ThermalSafetyWrapper(ConstellationEnv())])
    
    # Normalize observations and rewards
    # Normalizing Obs is crucial for PPO
    # Normalizing Reward? Maybe, but our reward is somewhat grounded in revenue. 
    # Let's normalize both for PPO stability.
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=log_dir,
        device="auto"
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path='models_constellation/',
        name_prefix='ppo_constellation'
    )
    
    # Logging callback
    log_callback = ConstellationLogCallback()
    
    print("Starting training for 10k steps (pilot)...")
    model.learn(
        total_timesteps=10_000,
        callback=[checkpoint_callback, log_callback],
        progress_bar=True
    )
    
    model.save("models_constellation/final_model")
    env.save("models_constellation/vec_normalize.pkl")
    print("Training complete.")

if __name__ == "__main__":
    main()
