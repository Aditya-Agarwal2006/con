"""
Evaluation + plotting for the curriculum-trained satellite controller.

Automatically detects which model is available (stage1 or stage2) and
plots thermal trace, compute throttle with duty cycle analysis, and
(for stage2) debris proximity.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sat_env import SatEnv, T_SOFT_LIMIT, T_HARD_LIMIT, T_TARGET

# --- MODEL SELECTION ---
# Try stage2 first, fall back to stage1
MODELS = [
    ("stage2_full", "full"),
    ("stage1_thermal", "thermal_only"),
    ("ppo_sat_v2", "full"),  # Legacy fallback
]

model = None
phase = None
for model_name, ph in MODELS:
    model_path = f"models/{model_name}"
    vecnorm_path = f"models/{model_name}_vecnormalize.pkl"
    if os.path.exists(model_path + ".zip") and os.path.exists(vecnorm_path):
        print(f"Loading model: {model_name} (phase={ph})")
        base_env = DummyVecEnv([lambda p=ph: SatEnv(phase=p)])
        env = VecNormalize.load(vecnorm_path, base_env)
        env.training = False
        env.norm_reward = False
        model = PPO.load(model_path)
        raw_env = env.venv.envs[0]
        phase = ph
        break

if model is None:
    print("No trained model found! Run training first:")
    print("  python train_curriculum.py --stage 1")
    sys.exit(1)

print("Running evaluation episode...")

# --- RUN EPISODE ---
obs = env.reset()
done = False
ended_by_death = False

temps = []
throttles = []
demands = []
distances = []
temp_rates = []

STEPS = 1000
for step in range(STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done_arr, info = env.step(action)
    done = bool(done_arr[0])

    temps.append(float(raw_env.current_temp))
    demands.append(float(raw_env.global_demand))
    temp_rates.append(float(raw_env.temp_rate))

    # Use the EMA-smoothed throttle (actual applied value), not raw action
    throttles.append(float(raw_env.prev_compute_throttle))
    if phase == "full":
        distances.append(float(raw_env.min_dist_km))

    if done:
        # Distinguish actual failure (thermal/collision) from natural truncation
        if step < STEPS - 2:  # Early termination = actual failure
            ended_by_death = True
            print(f"Episode FAILED at step {step} (thermal limit exceeded)")
        else:
            print(f"Episode completed ({step + 1} steps)")
        break

# --- ANALYSIS ---
DT = 120  # seconds per step
time_hours = np.arange(len(temps)) * DT / 3600.0  # Convert to hours

# Duty cycle detection: count transitions between high (>0.5) and low (<0.3) throttle
throttle_arr = np.array(throttles)
high_mask = throttle_arr > 0.5
transitions = np.sum(np.abs(np.diff(high_mask.astype(int))))
duty_cycles = transitions / 2.0  # Each cycle = one high→low→high
avg_throttle = float(np.mean(throttles))
temp_std = float(np.std(temps))
temp_mean = float(np.mean(temps))

# --- PLOTTING ---
n_plots = 4 if phase == "full" else 3
fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)
ax_idx = 0

# --- Plot 1: Thermal Trace ---
ax = axes[ax_idx]; ax_idx += 1
ax.plot(time_hours, temps, color='#d62728', linewidth=1.5, label='Temperature (K)')
ax.axhline(y=T_HARD_LIMIT, color='black', linestyle='--', linewidth=1.5, label=f'Hard Limit ({T_HARD_LIMIT:.0f}K)')
ax.axhline(y=T_SOFT_LIMIT, color='gray', linestyle=':', label=f'Soft Limit ({T_SOFT_LIMIT:.0f}K)')
ax.axhline(y=T_TARGET, color='#2ca02c', linestyle='-.', alpha=0.7, label=f'Target ({T_TARGET:.0f}K)')
ax.axhspan(T_TARGET - 8, T_TARGET + 8, alpha=0.08, color='green', label='Target Band (±8K)')
ax.set_ylabel('Temperature (K)', fontsize=12)
ax.set_title(f'Thermal Regulation — σ={temp_std:.1f}K, μ={temp_mean:.1f}K', fontsize=14)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# --- Plot 2: Compute Throttle + Demand ---
ax = axes[ax_idx]; ax_idx += 1
ax.fill_between(time_hours, throttles, color='#2ca02c', alpha=0.5, label='Compute Throttle')
ax.plot(time_hours, throttles, color='#186A18', linewidth=0.8)
ax.plot(time_hours, demands, 'k--', linewidth=1.5, alpha=0.7, label='Global Demand')
ax.set_ylabel('Throttle (0–1)', fontsize=12)
ax.set_ylim(-0.05, 1.1)
ax.set_title(f'Compute Output — Avg={avg_throttle*100:.1f}%, Duty Cycles={duty_cycles:.0f}', fontsize=14)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# --- Plot 3: Temperature Rate (pulsing quality) ---
ax = axes[ax_idx]; ax_idx += 1
colors = ['#2ca02c' if r <= 0 else '#d62728' for r in temp_rates]
ax.bar(time_hours, temp_rates, width=1.0/60.0, color=colors, alpha=0.6)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('dT/step (K)', fontsize=12)
ax.set_title('Temperature Rate — Green=Cooling, Red=Heating', fontsize=14)
ax.grid(True, alpha=0.3)

# --- Plot 4: Debris (full phase only) ---
if phase == "full":
    ax = axes[ax_idx]; ax_idx += 1
    ax.plot(time_hours[:len(distances)], distances, color='#1f77b4', linewidth=1.5, label='Nearest Debris')
    ax.axhline(y=10, color='red', linestyle='--', linewidth=1.5, label='Collision Zone (<10km)')
    ax.set_ylabel('Distance (km)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Debris Proximity Monitor', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Mission Time (Hours)', fontsize=12)
plt.tight_layout()
plt.savefig('paper_results.png', dpi=300)
plt.show()

# --- SUMMARY ---
print("=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"  Phase:                {phase}")
print(f"  Steps completed:      {len(temps)}/{STEPS}")
print(f"  Terminated early:     {'YES — FAILURE' if ended_by_death else 'NO — SURVIVED'}")
print(f"  Avg Compute:          {avg_throttle*100:.1f}%")
print(f"  Temperature Mean:     {temp_mean:.1f} K (target: {T_TARGET:.0f} K)")
print(f"  Temperature Std:      {temp_std:.1f} K {'✓' if temp_std < 8 else '✗ (>8K)'}")
print(f"  Peak Temperature:     {np.max(temps):.1f} K {'✓' if np.max(temps) < T_HARD_LIMIT else '✗ HIT HARD LIMIT'}")
print(f"  Duty Cycles:          {duty_cycles:.0f} (transitions={transitions})")
print(f"  Pulsing Quality:      {'GOOD — agent is cycling' if duty_cycles > 3 else 'POOR — monotonic behavior'}")
print("=" * 50)