import numpy as np
from collections import deque

class PIDController:
    """
    Classical Proportional-Integral-Derivative (PID) controller for thermal regulation.
    target_temp: The setpoint temperature (Kelvin)
    kp: Proportional gain
    ki: Integral gain
    kd: Derivative gain
    limit_output: Tuple (min, max) for the output signal (throttle)
    """
    def __init__(self, target_temp=350.0, kp=0.1, ki=0.001, kd=0.5, limit_output=(0.0, 1.0)):
        self.target_temp = target_temp
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out, self.max_out = limit_output
        self.prev_error = 0.0
        self.integral = 0.0

    def compute_action(self, current_temp, dt=1.0):
        error = self.target_temp - current_temp
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        # Anti-windup clamping
        self.integral = max(min(self.integral, 500.0), -500.0) 
        i_term = self.ki * self.integral
        
        # Derivative term
        d_error = (error - self.prev_error) / dt
        d_term = self.kd * d_error
        
        # Total output
        output = p_term + i_term + d_term
        
        # Save state
        self.prev_error = error
        
        # Clamp output to valid throttle range
        return max(min(output, self.max_out), self.min_out)

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0


class BangBangController:
    """
    Simple hysteresis (thermostat) controller.
    target_temp: The setpoint temperature
    hysteresis: The buffer band (e.g. +/- 1.0 degree)
    """
    def __init__(self, target_temp=350.0, hysteresis=2.0):
        self.target_temp = target_temp
        self.hysteresis = hysteresis
        self.state = 0.0  # Current throttle (0 or 1)

    def compute_action(self, current_temp):
        # If too hot, turn OFF
        if current_temp > self.target_temp + self.hysteresis:
            self.state = 0.0
        # If too cold, turn ON
        elif current_temp < self.target_temp - self.hysteresis:
            self.state = 1.0
        
        # Inside the hysteresis band, keep previous state to avoid chatter
        return self.state

    def reset(self):
        self.state = 0.0
