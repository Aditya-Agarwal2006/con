
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sat_env import SatEnv
from baselines import PIDController, BangBangController

def run_evaluation(controller_type, env, model=None, steps=1000):
    """
    Run a single evaluation episode with the specified controller.
    controller_type: 'RL', 'PID', or 'BangBang'
    """
    # VecEnv reset returns only obs
    obs = env.reset()
    terminated = False
    truncated = False
    
    temps = []
    throttles = []
    rewards = []
    
    # Instantiate classical controllers if needed
    if controller_type == 'PID':
        controller = PIDController(target_temp=350.0, kp=0.05, ki=0.0001, kd=0.5)
    elif controller_type == 'BangBang':
        controller = BangBangController(target_temp=350.0, hysteresis=2.0)
    
    current_step = 0
    while not (terminated or truncated) and current_step < steps:
        if controller_type == 'RL':
            # RL action
            action, _ = model.predict(obs, deterministic=True)
            # Clip action to valid range (0, 1) if using raw
            action = np.clip(action, 0.0, 1.0)
        else:
            # Classical action
            # The environment step expects an array
            # Extract temp from observation. 
            # We must access raw_env.current_temp directly.
            
            # The 'env' passed here is a VecEnv. We need to get the underlying env.
            raw_env = env.envs[0]
            current_temp = raw_env.current_temp
            
            if controller_type == 'PID':
                action_val = controller.compute_action(current_temp)
            elif controller_type == 'BangBang':
                action_val = controller.compute_action(current_temp)
            
            # Action must be shape (n_envs, action_dim) which is (1, 1)
            action = np.array([[action_val]], dtype=np.float32)

        # Step
        obs, reward, dones, info = env.step(action)
        terminated = dones[0] # VecEnv returns list of dones
        # Check truncation manually since VecEnv merges them
        if current_step >= steps - 1:
            truncated = True
            
        # Logging
        raw_env = env.envs[0]
        temps.append(raw_env.current_temp)
        # Handle case where prev_compute_throttle might be a scalar or array
        throttle_val = float(raw_env.prev_compute_throttle)
        throttles.append(throttle_val) # Log applied throttle
        rewards.append(reward[0])
        
        current_step += 1
        
    return temps, throttles, rewards

def main():
    # 1. Setup Environment
    # equivalent to: env = SatEnv(phase="thermal_only")
    env = DummyVecEnv([lambda: SatEnv(phase="thermal_only")])
    
    # Load normalization stats
    stats_path = "models/stage1_thermal_vecnormalize.pkl"
    env = VecNormalize.load(stats_path, env)
    env.training = False # Don't update stats during eval
    env.norm_reward = False # valid for PPO, allows us to see raw rewards? No, PPO expects normalized. 
    # Actually for evaluation plot we want intuitive rewards, but the model needs normalized obs.
    
    # 2. Load RL Model
    model = PPO.load("models/stage1_thermal", env=env)
    
    # 3. Use the same seed for all
    SEED = 42
    
    print("Running RL...")
    # env.seed(SEED) is deprecated. Set on underlying env.
    env.envs[0].reset(seed=SEED) 
    # Must reset the vec env wrapper too to propagate obs but seed is usually enough for underlying
    
    rl_temps, rl_throttles, rl_rewards = run_evaluation('RL', env, model)
    
    print("Running PID...")
    env.envs[0].reset(seed=SEED)
    pid_temps, pid_throttles, pid_rewards = run_evaluation('PID', env, model=None)
    
    print("Running Bang-Bang...")
    env.envs[0].reset(seed=SEED)
    bb_temps, bb_throttles, bb_rewards = run_evaluation('BangBang', env, model=None)
    
    # 4. Analysis & Plotting
    time_hours = np.arange(len(rl_temps)) * 120.0 / 3600.0
    
    print(f"RL:  Avg Temp={np.mean(rl_temps):.1f}K, Std={np.std(rl_temps):.2f}K, Avg Throttle={np.mean(rl_throttles):.1%} ({np.sum(rl_throttles):.1f} units)")
    print(f"PID: Avg Temp={np.mean(pid_temps):.1f}K, Std={np.std(pid_temps):.2f}K, Avg Throttle={np.mean(pid_throttles):.1%} ({np.sum(pid_throttles):.1f} units)")
    print(f"BB:  Avg Temp={np.mean(bb_temps):.1f}K, Std={np.std(bb_temps):.2f}K, Avg Throttle={np.mean(bb_throttles):.1%} ({np.sum(bb_throttles):.1f} units)")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Reward/Compute (Proxy for utility)
    # We want to show Compute Throughput primarily
    
    # Temperature Plot
    ax1.plot(time_hours, rl_temps, label='RL Agent', color='blue', linewidth=2)
    ax1.plot(time_hours, pid_temps, label='PID Control', color='green', linestyle='--', alpha=0.8)
    ax1.plot(time_hours, bb_temps, label='Bang-Bang', color='red', linestyle=':', alpha=0.8)
    
    ax1.axhline(350.0, color='gray', linestyle='-.', label='Target (350K)', alpha=0.5)
    ax1.axhline(358.0, color='orange', linestyle=':', label='Soft Limit (358K)')
    ax1.axhline(368.0, color='black', linestyle='--', label='Hard Limit (368K)')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('Thermal Regulation Comparison')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Throttle Plot
    ax2.plot(time_hours, rl_throttles, label='RL Throttle', color='blue', linewidth=1.5, alpha=0.9)
    ax2.plot(time_hours, pid_throttles, label='PID Throttle', color='green', linestyle='--', alpha=0.7)
    ax2.plot(time_hours, bb_throttles, label='Bang-Bang', color='red', linestyle=':', alpha=0.5)
    
    # Overlay Eclipse regions if possible (need to get is_lit from env)
    # Since we can't easily extract that history from the VecEnv wrapper without logging it inside loop
    # We'll skip shading for now or reconstruct it.
    
    ax2.set_ylabel('Compute Throttle (0-1)')
    ax2.set_xlabel('Mission Time (Hours)')
    ax2.set_title(f'Compute Output | RL Profit: {np.sum(rl_throttles):.0f} vs PID: {np.sum(pid_throttles):.0f}')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('baseline_comparison.png', dpi=300)
    print("Comparison plot saved to baseline_comparison.png")

if __name__ == "__main__":
    main()
