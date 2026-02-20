import gymnasium as gym
import numpy as np

class ThermalSafetyWrapper(gym.Wrapper):
    """
    Safe RL formalization: A non-learnable safety shield.
    
    This wrapper intercepts the RL agent's action (load distribution logits) 
    and checks the current thermal state of the constellation. If any satellite
    exceeds the soft limit, the filter forces its action logit to a large negative 
    number, effectively reducing its load assignment (via Softmax) to zero.
    
    The agent trains *through* this wrapper, learning that trying to send
    load to overheating satellites results in zero throughput.
    """
    def __init__(self, env, max_temp=358.0):
        super(ThermalSafetyWrapper, self).__init__(env)
        self.max_temp = max_temp

    def step(self, action):
        # Action is expected to be [n_sats] continuous logits
        safe_action = np.copy(action)
        
        # We assume the underlying env has a 'sats' attribute (ConstellationEnv)
        # In a real deployed system, this would query telemetry.
        for i, sat in enumerate(self.env.unwrapped.sats):
            if sat.current_temp >= self.max_temp:
                # Force logit to -100.0 so Softmax weight -> 0
                safe_action[i] = -100.0
                
        return self.env.step(safe_action)
