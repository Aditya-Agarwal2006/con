import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from constellation_env import ConstellationEnv
from safety_filter import ThermalSafetyWrapper

def main():
    print("Loading environment and model for visualization...")
    env_fn = lambda: ThermalSafetyWrapper(ConstellationEnv())
    env = DummyVecEnv([env_fn])
    env = VecNormalize.load("models_constellation/vec_normalize.pkl", env)
    env.training = False
    
    model = PPO.load("models_constellation/final_model", env=env)
    
    obs = env.reset()
    
    # Pre-compute data by running an episode
    print("Simulating 300 steps (10-hour orbit window)...")
    steps = 300
    all_temps = []
    all_throttles = []
    all_sunlit = []
    all_demands = []
    
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        info = infos[0]
        raw_env = env.venv.envs[0] # Bypass dummy and wrapper
        
        temps = [s.current_temp for s in raw_env.unwrapped.sats]
        throttles = [s.prev_compute_throttle for s in raw_env.unwrapped.sats]
        t = raw_env.unwrapped.sats[0].t_current
        sunlit = [1.0 if s.sat.at(t).is_sunlit(s.eph) else 0.0 for s in raw_env.unwrapped.sats]
        
        all_temps.append(temps)
        all_throttles.append(throttles)
        all_sunlit.append(sunlit)
        all_demands.append(info.get("global_demand_units", 0) / 24.0) # Average demand per sat
        
        if dones[0]:
            obs = env.reset()

    # Create Animation
    print("Rendering animation to orbit_dispatch.gif...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('Constellation Dispatcher: "Follow the Shade"', fontsize=16)
    
    # Circular plot setup
    n_sats = 24
    theta = np.linspace(0, 2*np.pi, n_sats, endpoint=False)
    
    # Ax1: Circular diagram representing orbits and load
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.axis('off')
    
    sun_dir = np.array([-1, 0]) # Sun comes from left
    
    # Ax2: Time series
    ax2.set_xlim(0, 50) # Scroll window
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Throt / Demand")
    ax2.set_xlabel("Time (steps)")
    
    line_demand, = ax2.plot([], [], 'k--', label="Global Demand")
    line_served, = ax2.plot([], [], 'b-', label="Served Load")
    ax2.legend(loc="upper right")
    
    scatter = ax1.scatter([], [], s=[], c=[], cmap='inferno', vmin=340, vmax=360, edgecolors='black', alpha=0.9)
    # Background circle
    circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle=':')
    ax1.add_patch(circle)
    
    def update(frame):
        # Frame data
        temps = np.array(all_temps[frame])
        throttles = np.array(all_throttles[frame])
        sunlit = np.array(all_sunlit[frame])
        
        # Position satellites in a circle. 
        # We roughly align them by ID, but rotate over time to simulate orbiting.
        # Actually, walker delta isn't a single ring, but we project it to 2D for visual mapping.
        orbit_angle = frame * (2 * np.pi / 200) # Full rotation every 200 steps
        current_theta = theta + orbit_angle
        x = np.cos(current_theta)
        y = np.sin(current_theta)
        
        # Shade representation (if sunlit==0, it's in the dark half)
        # We draw the scatter points
        sizes = 100 + (throttles * 500) # Size = compute load
        
        scatter.set_offsets(np.c_[x, y])
        scatter.set_array(temps)
        scatter.set_sizes(sizes)
        
        # Add a subtle shadow over the right half to indicate eclipse
        # To make it dynamic without complex patches, we just let the dots change color
        
        # Update line charts
        window_start = max(0, frame - 50)
        window = slice(window_start, frame)
        
        x_data = np.arange(window_start, frame)
        
        avg_served = np.mean(all_throttles[window_start:frame], axis=1) if frame > 0 else []
        dem = all_demands[window_start:frame]
        
        line_demand.set_data(x_data, dem)
        line_served.set_data(x_data, avg_served)
        
        ax2.set_xlim(window_start, window_start + 50)
        
        # Update Title with Stats
        current_demand = all_demands[frame] * 24
        current_served = np.sum(throttles)
        avg_t = np.mean(temps)
        ax1.set_title(f"Step {frame} | Demand: {current_demand:.1f} | Served: {current_served:.1f} | Fleet Avg T: {avg_t:.1f}K")
        
        return scatter, line_demand, line_served

    ani = animation.FuncAnimation(fig, update, frames=min(steps, 200), interval=50, blit=False)
    
    # Save as gif using pillow
    ani.save('orbit_dispatch.gif', writer='pillow', fps=20)
    print("Video saved as orbit_dispatch.gif")

if __name__ == "__main__":
    main()
