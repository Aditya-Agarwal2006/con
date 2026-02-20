
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sat_env import SatEnv, T_TARGET, T_SOFT_LIMIT, T_HARD_LIMIT
import orbital_mechanics

class ConstellationEnv(gym.Env):
    """
    Centralized Dispatcher Environment for a 24-satellite constellation.
    
    The agent acts as a global load balancer, receiving global compute demand
    and distributing it across 24 satellites.
    
    State:
      - Global Demand (1)
      - Per Sat: [Temp (norm), Is_Lit (0/1), Orbit_Phase (cos/sin)]
    
    Action:
      - Load weights (24): Softmax is applied to these to distribute demand.
    """
    def __init__(self):
        super(ConstellationEnv, self).__init__()
        
        self.n_sats = 24
        
        # 1. Generate Constellation
        print("Generating Walker Delta Constellation (24/3/1)...")
        self.tles = orbital_mechanics.generate_walker_delta(n_sats=self.n_sats)
        
        # 2. Instantiate Satellites
        print("Initializing 24 Satellite Environments (Phase=full)...")
        self.sats = []
        for i in range(self.n_sats):
            env = SatEnv(phase="full", tle_input=self.tles[i])
            self.sats.append(env)
            
        # 3. 120D Action Space (24 sats * 5 actions) 
        # [thrust_r, thrust_t, thrust_n, compute_throttle, attitude_theta]
        single_low = np.array([-0.1, -0.1, -0.1, 0.0, 0.0])
        single_high = np.array([0.1, 0.1, 0.1, 1.0, np.pi])
        self.action_space = spaces.Box(
            low=np.tile(single_low, self.n_sats),
            high=np.tile(single_high, self.n_sats),
            dtype=np.float32
        )
        
        # 4. Coverage Grid Precomputation (72x36)
        lons = np.linspace(-np.pi, np.pi, 72, endpoint=False)
        lats = np.linspace(-np.pi/2, np.pi/2, 36, endpoint=False)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        self.cell_vectors = np.vstack([
            (np.cos(lat_grid) * np.cos(lon_grid)).flatten(),
            (np.cos(lat_grid) * np.sin(lon_grid)).flatten(),
            np.sin(lat_grid).flatten()
        ]) # (3, 2592)
        
        # Target region mask: Latitudes +/- 60 deg
        self.target_mask = (lat_grid.flatten() >= -1.047) & (lat_grid.flatten() <= 1.047)
        self.n_target_cells = np.sum(self.target_mask)
        self.coverage_map = np.zeros(2592)
        
        # 5. Observation Space
        # 1 (Demand) + 24*15 (Sat states) + 2592 (Coverage map) = 2953
        self.obs_dim = 1 + (self.n_sats * 15) + 2592
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        self.global_demand = 0.6
        self.steps = 0
        self.max_steps = 1000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        
        # Global demand init
        self.global_demand = 0.6
        self.demand_phase = np.random.uniform(0.0, 2.0 * np.pi)
        
        # Reset all satellites
        # We need to ensure they are synchronized in time?
        # SatEnv.reset() sets time to self.ts.now()
        # If we call them sequentially, they might have micro-second differences, 
        # but for orbital physics at 120s step, that's negligible.
        
        sat_obs_list = []
        for i, env in enumerate(self.sats):
            # Seed each slightly differently so they don't have identical internal random states
            s_seed = seed + i if seed is not None else None
            _ = env.reset(seed=s_seed)
            
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        
        # 1. Update Global Demand
        base_demand = 0.55 + 0.25 * np.sin((2.0 * np.pi * self.steps / 300.0) + self.demand_phase)
        self.global_demand = float(np.clip(base_demand, 0.2, 0.9))
        
        # 2. Reshape 120D action into 24 x 5D chunks
        actions = action.reshape((self.n_sats, 5))
        
        total_sat_reward = 0.0
        terminated = False
        truncated = (self.steps >= self.max_steps)
        info = {
            "avg_temp": [],
            "total_throughput": 0.0,
            "failed_sats": 0
        }
        
        sat_positions = np.zeros((self.n_sats, 3))
        
        # 3. Step all satellites
        for i, env in enumerate(self.sats):
            sat_action = np.array(actions[i], dtype=np.float32)
            _, r, term, _, sat_info = env.step(sat_action)
            
            total_sat_reward += r
            if term:
                terminated = True
                info["failed_sats"] += 1
                
            info["avg_temp"].append(env.current_temp)
            info["total_throughput"] += sat_info["compute_throttle"]
            
            # Extract position unit vector for coverage matrix
            pos_km = env.sat.at(env.t_current).position.km
            norm = np.linalg.norm(pos_km)
            if norm > 0 and not np.isnan(norm):
                sat_positions[i] = pos_km / norm
            else:
                sat_positions[i] = np.array([1.0, 0.0, 0.0]) # Fallback pointing if numerical error

        # 4. Coverage Matrix Calculation ($R_{coverage}$)
        # Compute angles between all satellites and all earth grid cells
        dot_products = sat_positions @ self.cell_vectors # (24, 2592)
        # Visibility threshold: angle < 25.7 deg -> cos(angle) > 0.901
        visible_matrix = dot_products > 0.901
        
        self.coverage_map = np.sum(visible_matrix, axis=0) # Number of sats seeing each cell
        covered_target_cells = np.sum((self.coverage_map >= 2) & self.target_mask)
        r_coverage = covered_target_cells / self.n_target_cells
        
        # 5. Global Multi-Objective Synthesis
        # W_1 is Coverage. W_2,3,4,5 are already mapped in SatEnv's local reward.
        w_coverage = 10.0
        total_reward = (w_coverage * r_coverage) + (total_sat_reward / self.n_sats)
        
        info["avg_temp"] = float(np.mean(info["avg_temp"]))
        info["coverage_fraction"] = r_coverage
        
        return self._get_obs(), total_reward, terminated, truncated, info
        
        return self._get_obs(), total_reward, terminated, truncated, info

    def _get_obs(self):
        obs = [self.global_demand]
        for env in self.sats:
            obs.extend(env._get_obs().tolist())
            
        obs.extend(self.coverage_map.tolist())
        return np.array(obs, dtype=np.float32)

    def render(self):
        pass

    def close(self):
        pass
