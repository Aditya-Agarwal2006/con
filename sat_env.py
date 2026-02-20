import gymnasium as gym
import numpy as np
import random
from collections import deque
from gymnasium import spaces
from skyfield.api import load, EarthSatellite
from scipy.constants import sigma
from scipy.spatial import cKDTree

# --- PHYSICS CONSTANTS (Starlink V2 + H100) ---
MASS = 800.0
C_P = 900.0
AREA_RAD = 1.6            # m² — with DT=120s: ~50-60% throttle sustainable, full throttle enters danger zone
AREA_CROSS = 3.0
ALPHA = 0.21
EPSILON = 0.85
Q_BUS_BASE = 200.0
Q_GPU_PEAK = 700.0
Q_THRUST_PEAK = 500.0
T_SOFT_LIMIT = 358.0
T_HARD_LIMIT = 368.0
T_TARGET = 350.0           # ~60% throttle equilibrium temperature (AREA_RAD=1.6)
SOLAR_CONSTANT = 1361

# Rolling window size for stability metrics
STABILITY_WINDOW = 30


class SatEnv(gym.Env):
    """
    Satellite orbital datacenter environment.

    Phases:
      - "thermal_only": simplified env for learning thermal pulsing
                        (no debris penalties, reduced fuel pressure, 2D action)
      - "full":         complete env with debris, fuel, and thrust
    """

    def __init__(self, phase="thermal_only", tle_input=None):
        super(SatEnv, self).__init__()
        self.phase = phase
        self.tle_input = tle_input
        self.MAX_FUEL = 50.0
        self.fuel = self.MAX_FUEL

        # --- ACTION SPACE ---
        if self.phase == "thermal_only":
            # [compute_throttle, attitude_theta]
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0]),
                high=np.array([1.0, np.pi]),
                dtype=np.float32,
            )
        else:
            # Full: [thrust_r, thrust_t, thrust_n, compute_throttle, attitude_theta]
            self.action_space = spaces.Box(
                low=np.array([-0.1, -0.1, -0.1, 0.0, 0.0]),
                high=np.array([0.1, 0.1, 0.1, 1.0, np.pi]),
                dtype=np.float32,
            )

        # --- OBSERVATION SPACE ---
        if self.phase == "thermal_only":
            # [pos(3), vel(3), temp, temp_rate, is_lit, demand, avg_compute, temp_trend, theta]
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
            )
        else:
            # Full: [pos(3), vel(3), temp, temp_rate, is_lit, dist, demand, fuel, avg_compute, temp_trend, theta]
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
            )

        # --- CATALOG LOADER ---
        print(f"Loading catalog (phase={self.phase})...")
        if self.phase == "full":
            try:
                self.debris_satellites = load.tle_file("iridium-33-debris.txt")
                self.active_satellites = load.tle_file("active.txt")
                self.full_catalog = self.debris_satellites + self.active_satellites
                # Optimization for RL: Truncate to 100 objects to maintain fast step times (PDD Week 1)
                self.full_catalog = self.full_catalog[:100]
            except Exception:
                print("Warning: Full catalog not found. Using subset.")
                self.full_catalog = load.tle_file("iridium-33-debris.txt")[:100]
            print(f"Tracking {len(self.full_catalog)} objects.")
        else:
            self.full_catalog = []
            print("Thermal-only mode: debris tracking disabled.")

        self.ts = load.timescale()
        self.eph = load("de421.bsp")
        self.current_threats = []
        self.min_dist_km = 1e6
        self.temp_rate = 0.0
        self.prev_compute_throttle = 0.0

        # Rolling history for stability reward
        self.temp_history = deque(maxlen=STABILITY_WINDOW)
        self.compute_history = deque(maxlen=STABILITY_WINDOW)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t_current = self.ts.now()
        self.steps = 0
        self.fuel = self.MAX_FUEL

        # Start near target temperature
        self.current_temp = random.uniform(345.0, 355.0)
        self.temp_rate = 0.0
        self.prev_compute_throttle = 0.0
        self.current_theta = 0.0

        # Clear rolling histories
        self.temp_history.clear()
        self.compute_history.clear()

        # Global demand — phase-shifted sinusoid
        self.demand_phase = random.uniform(0.0, 2.0 * np.pi)
        self.global_demand = 0.6

        # Agent orbit
        if self.tle_input:
            line1, line2 = self.tle_input
            self.sat = EarthSatellite(line1, line2, "Agent", self.ts)
        else:
            # Default 780 km polar — high-risk belt
            line1 = "1 70001U 24001A   24040.12345678  .00012345  00000-0  12345-3 0  9999"
            line2 = "2 70001  86.4000 120.0000 0001000 000.0000 180.0000 14.30000000    19"
            self.sat = EarthSatellite(line1, line2, "Agent", self.ts)

        # Build KDTree for debris tracking (full phase only)
        if self.phase == "full" and len(self.full_catalog) > 0:
            agent_pos = self.sat.at(self.t_current).position.km
            valid_indices = []
            valid_positions = []
            for i, obj in enumerate(self.full_catalog):
                try:
                    pos = np.asarray(obj.at(self.t_current).position.km, dtype=np.float64)
                except Exception:
                    continue
                if pos.shape == (3,) and np.all(np.isfinite(pos)):
                    valid_indices.append(i)
                    valid_positions.append(pos)

            if len(valid_positions) > 0:
                tree = cKDTree(np.vstack(valid_positions))
                k = min(250, len(valid_positions))
                _, idxs = tree.query(agent_pos, k=k)
                idxs = np.atleast_1d(idxs).astype(int)
                self.current_threats = [self.full_catalog[valid_indices[j]] for j in idxs]
            else:
                self.current_threats = []
        else:
            self.current_threats = []

        self.min_dist_km = 1e6

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1

        # --- PARSE ACTIONS ---
        if self.phase == "thermal_only":
            thrust_vec = np.zeros(3)
            thrust_mag = 0.0
            fuel_cost = 0.0
            raw_throttle = float(np.clip(action[0], 0.0, 1.0))
            self.current_theta = float(np.clip(action[1], 0.0, np.pi)) if len(action) > 1 else 0.0
        else:
            thrust_vec = np.array(action[0:3], dtype=np.float64) # [-0.1, 0.1]
            thrust_mag = float(np.linalg.norm(thrust_vec))
            fuel_cost = thrust_mag # 1 unit deltaV = 1 unit fuel
            if self.fuel > 0.0:
                self.fuel = max(0.0, self.fuel - fuel_cost)
            else:
                thrust_vec[:] = 0.0
                thrust_mag = 0.0
                fuel_cost = 0.0
            raw_throttle = float(np.clip(action[3], 0.0, 1.0))
            self.current_theta = float(np.clip(action[4], 0.0, np.pi))

        # EMA smoothing: structurally enforces smooth compute output.
        # The agent can't make sudden jumps — max change per step ≈ alpha * gap.
        # This is key for fleet operations: each satellite provides steady output.
        EMA_ALPHA = 0.05  # Very smooth (ramp 0→1 takes ~60 steps = 2hrs)
        compute_throttle = (1 - EMA_ALPHA) * self.prev_compute_throttle + EMA_ALPHA * raw_throttle

        # --- PHYSICS ---
        dt_min = 1.0
        self.t_current = self.ts.utc(
            self.t_current.utc_datetime().year,
            self.t_current.utc_datetime().month,
            self.t_current.utc_datetime().day,
            self.t_current.utc_datetime().hour,
            self.t_current.utc_datetime().minute + dt_min,
        )

        is_lit = self.sat.at(self.t_current).is_sunlit(self.eph)
        # Solar aspect angle factor: cos(theta). We use abs() so both flat sides absorb equally.
        # theta = 0 -> full broadside. theta = pi/2 -> edge on.
        theta_factor = abs(np.cos(self.current_theta))
        q_solar = (AREA_CROSS * SOLAR_CONSTANT * ALPHA * theta_factor) if is_lit else 0.0
        q_earth = 230.0 * 0.2 * EPSILON * AREA_RAD
        q_thrust = thrust_mag * Q_THRUST_PEAK
        q_gen = Q_BUS_BASE + (compute_throttle * Q_GPU_PEAK) + q_thrust
        q_out = EPSILON * sigma * AREA_RAD * (self.current_temp ** 4)

        prev_temp = self.current_temp
        self.current_temp += ((q_solar + q_earth + q_gen - q_out) / (MASS * C_P)) * 120.0  # 2-minute steps for faster thermal response
        self.temp_rate = self.current_temp - prev_temp

        # Update rolling histories
        self.temp_history.append(self.current_temp)
        self.compute_history.append(compute_throttle)

        # Update demand
        self.global_demand = float(
            np.clip(
                0.55 + 0.25 * np.sin((2.0 * np.pi * self.steps / 180.0) + self.demand_phase),
                0.25, 0.85,
            )
        )

        # --- DEBRIS (full phase only) ---
        if self.phase == "full":
            pos_agent = self.sat.at(self.t_current).position.km
            min_dist_km = 1e6
            for threat in self.current_threats:
                try:
                    pos_threat = np.asarray(threat.at(self.t_current).position.km, dtype=np.float64)
                except Exception:
                    continue
                if pos_threat.shape != (3,) or not np.all(np.isfinite(pos_threat)):
                    continue
                dist = float(np.linalg.norm(pos_agent - pos_threat))
                if dist < min_dist_km:
                    min_dist_km = dist
            self.min_dist_km = min_dist_km

        # =============================================
        #     PDD MULTI-OBJECTIVE REWARD
        # =============================================
        terminated = False
        
        # 1. Collision Penalty (R_collision)
        # PDD: <1km: 1e-3, <5km: 1e-4, <10km: 1e-5
        r_collision = 0.0
        if self.phase == "full":
            if self.min_dist_km < 1.0:
                r_collision = -1e-3
                terminated = True
            elif self.min_dist_km < 5.0:
                r_collision = -1e-4
            elif self.min_dist_km < 10.0:
                r_collision = -1e-5
                
        # 2. Thermal Management (R_thermal)
        # Smooth gaussian around T_opt (313 K / 40 C) with sigma 20
        T_opt = 313.0
        r_thermal = float(np.exp(- ((self.current_temp - T_opt)**2) / (2 * 20.0**2)))
        
        # 3. Compute Uptime (R_compute)
        # Revenue only granted if below thermal hard limit.
        if self.current_temp < T_HARD_LIMIT:
            r_compute = compute_throttle
        else:
            r_compute = 0.0
            
        # 4. Fuel Efficiency (R_fuel)
        # Normalized by max potential delta V per step (~0.173)
        r_fuel = thrust_mag / 0.173 if self.phase == "full" else 0.0
        
        # Compute Weighted Local Reward
        # Fix for PDD logic: w_2 needs to be massive to outscale w_1=10
        w_collision = 100000.0 # 100,000 * 1e-3 = 100.0 penalty
        w_thermal = 5.0
        w_compute = 8.0
        w_fuel = 1.0
        
        reward = (w_collision * r_collision) + (w_thermal * r_thermal) + (w_compute * r_compute) - (w_fuel * r_fuel)
        
        # Hard limits & Survival
        if self.current_temp >= T_HARD_LIMIT:
            reward -= 50.0 # Catastrophic failure
            terminated = True
            
        if not terminated:
            reward += 0.1 # Small survival baseline
            
        reward = float(np.clip(reward, -100.0, 15.0))
        self.prev_compute_throttle = compute_throttle

        # Truncate
        truncated = self.steps >= 1000

        # --- INFO DICT (for TensorBoard diagnostics) ---
        info = {
            "temperature": self.current_temp,
            "temp_rate": self.temp_rate,
            "compute_throttle": compute_throttle,
            "global_demand": self.global_demand,
            "fuel": self.fuel,
            "min_dist_km": self.min_dist_km,
        }
        if len(self.temp_history) >= STABILITY_WINDOW:
            info["temp_std"] = float(np.std(self.temp_history))
            info["temp_mean"] = float(np.mean(self.temp_history))
            info["avg_compute"] = float(np.mean(self.compute_history))

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        pos = self.sat.at(self.t_current).position.km / 6371.0
        vel = self.sat.at(self.t_current).velocity.km_per_s / 7.5
        temp_norm = (self.current_temp - 200) / 200.0
        temp_rate_norm = float(np.clip(self.temp_rate / 5.0, -1.0, 1.0))
        is_lit = 1.0 if self.sat.at(self.t_current).is_sunlit(self.eph) else 0.0

        # Rolling averages as obs features (give the agent memory of recent behavior)
        avg_compute = float(np.mean(self.compute_history)) if len(self.compute_history) > 0 else 0.0
        temp_trend = float(np.mean(list(self.temp_history)[-10:]) - np.mean(list(self.temp_history)[:10])) \
            if len(self.temp_history) >= 20 else 0.0
        temp_trend_norm = float(np.clip(temp_trend / 10.0, -1.0, 1.0))

        # Normalize theta for obs
        theta_norm = self.current_theta / np.pi

        if self.phase == "thermal_only":
            obs = np.concatenate(
                (pos, vel, [temp_norm, temp_rate_norm, is_lit,
                            self.global_demand, avg_compute, temp_trend_norm, theta_norm])
            )
        else:
            dist_norm = min(self.min_dist_km, 1000.0) / 1000.0
            fuel_norm = self.fuel / self.MAX_FUEL
            obs = np.concatenate(
                (pos, vel, [temp_norm, temp_rate_norm, is_lit, dist_norm,
                            self.global_demand, fuel_norm, avg_compute, temp_trend_norm, theta_norm])
            )

        return obs.astype(np.float32)
