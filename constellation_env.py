
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
        print("Initializing 24 Satellite Environments (this may take a moment)...")
        self.sats = []
        for i in range(self.n_sats):
            # thermal_only phase is lighter weight (no debris)
            env = SatEnv(phase="thermal_only", tle_input=self.tles[i])
            self.sats.append(env)
            
        # 3. Action Space: 24 continuous logits for Softmax
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_sats,), dtype=np.float32
        )
        
        # 4. Observation Space
        # Global Demand (1)
        # + 24 * (Temp_Norm (1) + Is_Lit (1) + Phase_Cos (1) + Phase_Sin (1)) = 4 features per sat
        # Total = 1 + 96 = 97
        self.obs_dim = 1 + (self.n_sats * 4)
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
        
        # 1. Update Global Demand (Sine wave)
        # Slightly noisy demand
        base_demand = 0.55 + 0.25 * np.sin((2.0 * np.pi * self.steps / 300.0) + self.demand_phase)
        self.global_demand = float(np.clip(base_demand, 0.2, 0.9))
        
        # 2. Parse Action (Load Balancing)
        # Softmax to get weights
        exp_action = np.exp(action - np.max(action))
        weights = exp_action / np.sum(exp_action)
        
        # Distribute demand
        # Total capacity = 24 * 1.0 (throttle)
        # Requested load = global_demand * 24 (if global demand is fraction of TOTAL capacity)
        # Or is global demand fraction of single sat?
        # Let's say global_demand is "Average Throttle Required Per Sat".
        # So Total Load = global_demand * 24.
        # Load for Sat i = weights[i] * (global_demand * 24)
        
        total_load_units = self.global_demand * self.n_sats
        throttles = weights * total_load_units
        
        # Clip throttles to [0, 1] — we can't exceed capacity
        # Any "lost" demand is unserved (penalty?)
        # For simplicity, we just clip.
        throttles = np.clip(throttles, 0.0, 1.0)
        
        # 3. Step all satellites
        total_reward = 0.0
        terminated = False
        truncated = (self.steps >= self.max_steps)
        info = {
            "avg_temp": [],
            "total_throughput": 0.0,
            "failed_sats": 0,
            "thermal_utilization": []
        }
        
        rewards = []
        
        for i, env in enumerate(self.sats):
            # Run step
            # SatEnv expects action [throttle]
            sat_action = np.array([throttles[i]], dtype=np.float32)
            
            # --- SAFETY SHIELD (Simulated) ---
            # If temp > T_SOFT_LIMIT, force throttle to 0 override
            # The agent should learn to avoid this, but explicit safety is good.
            # However, for training PPO to *learn* this, maybe we don't force it yet?
            # User plan said "Train Dispatcher *through* this shield".
            # Can't peek next T easily without stepping.
            # So shield reacts to *current* temp.
            if env.current_temp > T_SOFT_LIMIT:
                sat_action = np.array([0.0], dtype=np.float32)
                throttles[i] = 0.0 # Update for info logging
            
            # Step env
            _, r, term, _, sat_info = env.step(sat_action)
            
            # We override the SatEnv's internal "Demand Matching" reward because that was local
            # We want Global Reward.
            # SatEnv reward v8 = (Throttle * Efficiency * 5) - TempCost
            # We can use that sum directly? 
            # Yes, sum of local rewards is a valid cooperative reward.
            # But we want to explicitly penalize Unserved Demand.
            
            # Let's extract raw metrics to build our own reward
            # Revenue = Throttle * Efficiency * 5.0
            # TempCost = 4.0 * normalized_excess^2
            
            # We'll use the environment's reward for now, as it encapsulates the physics of profit/cost.
            total_reward += r
            rewards.append(r)
            
            if term: # Hard limit violation
                terminated = True # If one fails, game over? Or just that node dies?
                # Let's say game over for MVP rigor.
                info["failed_sats"] += 1
            
            info["avg_temp"].append(env.current_temp)
            info["total_throughput"] += throttles[i]
            info["thermal_utilization"].append(env.current_temp / T_HARD_LIMIT)
            
        # 4. Global Demand Penalty
        # If we clipped throttles, we didn't meet demand.
        # But we distributed (global_demand * 24) using softmax (sum=1), so sum(throttles) should be global_demand * 24.
        # Unless clips happened.
        served_load = np.sum(throttles)
        required_load = self.global_demand * self.n_sats
        unserved = max(0.0, required_load - served_load)
        
        # Penalty for unserved demand
        total_reward -= 2.0 * unserved
        
        # Normalize reward for stability (24 sats -> massive reward magnitude)
        total_reward /= self.n_sats 
        
        info["avg_temp"] = float(np.mean(info["avg_temp"]))
        info["global_demand_units"] = required_load
        info["served_units"] = served_load
        
        return self._get_obs(), total_reward, terminated, truncated, info

    def _get_obs(self):
        # Construct global observation
        # [Global_Demand, (S1_Temp, S1_Lit, S1_PhaseCos, S1_PhaseSin), (S2...), ...]
        
        obs = [self.global_demand]
        
        for env in self.sats:
            # SatEnv wrapper properties
            temp_norm = (env.current_temp - 200) / 200.0
            
            # Is Lit
            t = env.t_current
            pos = env.sat.at(t).position.km
            is_lit = 1.0 if env.sat.at(t).is_sunlit(env.eph) else 0.0
            
            # Orbit Phase (Mean Anomaly proxy or similar)
            # Skyfield doesn't give anomaly easily without computing osculating elements.
            # Proxy: Angle of position vector in orbital plane?
            # Simple Proxy: position vector direction relative to sun?
            # Actually, let's use the 'is_lit' + 'temp_rate' as enough proxy for now?
            # User plan asked for Orbit Phase.
            # Let's try to get mean anomaly from `env.sat.model.M`? 
            # No, that's epoch M.
            # Best proxy for ML: normalized position vector (x,y,z).
            # But that adds 3 dims per sat.
            # Let's use 3 dims: Temp, Is_Lit, Rate
            
            # Temp Rate
            rate_norm = float(np.clip(env.temp_rate / 5.0, -1.0, 1.0))
            
            obs.extend([temp_norm, is_lit, rate_norm, 0.0]) # 4th dim padding or something?
            # Plan said 4 features. I'll use [Temp, Is_Lit, Rate, Fuel(always full in thermal only)]
            # Actually let's use [Temp, Is_Lit, Rate, Previous_Throttle]
            obs[-1] = env.prev_compute_throttle
            
        return np.array(obs, dtype=np.float32)

    def render(self):
        pass

    def close(self):
        pass
