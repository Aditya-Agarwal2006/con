
import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from constellation_env import ConstellationEnv
from safety_filter import ThermalSafetyWrapper

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

class ConstellationFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom Deep Sets Extractor for the God Environment.
    Instead of a massive MLP that struggles to learn 2953 parameters from scratch,
    we use a permutation-invariant architecture (Deep Sets).
    1. A shared MLP processes each of the 24 satellite's 15 states identically.
    2. We max-pool across satellites to get a permutation-invariant fleet representation.
    3. We compress the sparse 2592-cell coverage map.
    4. We fuse Demand, Fleet Rep, and Coverage Rep into 256 features.
    """
    def __init__(self, observation_space, features_dim=256):
        super(ConstellationFeatureExtractor, self).__init__(observation_space, features_dim)
        
        self.n_sats = 24
        self.sat_feat = 15
        self.cov_cells = 2592
        
        # 1. Satellite Encoder (15 -> 64)
        self.sat_net = nn.Sequential(
            nn.Linear(self.sat_feat, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 2. Coverage map compressor (2592 -> 128)
        self.cov_net = nn.Sequential(
            nn.Linear(self.cov_cells, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        
        # 3. Final fusion (Demand(1) + SatAgg(64) + CovRep(128) = 193)
        self.fusion = nn.Sequential(
            nn.Linear(193, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations):
        b = observations.shape[0]
        
        demand = observations[:, 0:1] # (b, 1)
        
        sat_obs = observations[:, 1:361].view(b, self.n_sats, self.sat_feat) # (b, 24, 15)
        # Apply shared MLP across sats
        sat_rep = self.sat_net(sat_obs.view(b * self.n_sats, self.sat_feat))
        sat_rep = sat_rep.view(b, self.n_sats, 64)
        
        # Max-pool over satellites (permutation invariant aggregation)
        sat_agg, _ = torch.max(sat_rep, dim=1) # (b, 64)
        
        cov_map = observations[:, 361:] # (b, 2592)
        cov_rep = self.cov_net(cov_map) # (b, 128)
        
        fusion_in = torch.cat([demand, sat_agg, cov_rep], dim=1) # (b, 193)
        
        return self.fusion(fusion_in)

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
            if "total_throughput" in info:
                self.logger.record("constellation/total_throughput", info["total_throughput"])
            if "coverage_fraction" in info:
                self.logger.record("constellation/coverage_fraction", info["coverage_fraction"])
                
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
    policy_kwargs = dict(
        features_extractor_class=ConstellationFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    
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
        device="auto",
        policy_kwargs=policy_kwargs
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
