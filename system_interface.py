import numpy as np
import requests
import time
from abc import ABC, abstractmethod

class SystemInterface(ABC):
    @abstractmethod
    def get_state(self):
        """Get current system state"""
        pass
    
    @abstractmethod
    def apply_input(self, u):
        """Apply control input to system"""
        pass
    
    @abstractmethod
    def reset(self, x0):
        """Reset system to initial state"""
        pass

class SimulationInterface(SystemInterface):
    def __init__(self, system_config, mpc_config):
        self.system_config = system_config
        self.mpc_config = mpc_config
        self.x = mpc_config.initial_state.copy()
        self.dt = mpc_config.dt
        self.A = mpc_config.A
        self.B = mpc_config.B
        self.noise_level = system_config.sim_config['noise_level']
        self.disturbance_level = system_config.sim_config['disturbance_level']
    
    def get_state(self):
        return self.x.copy()
    
    def apply_input(self, u):
        # Apply input with noise and disturbance
        noise = np.random.normal(0, self.noise_level, self.x.shape)
        disturbance = np.random.normal(0, self.disturbance_level, self.x.shape)
        
        # Update state
        self.x = self.A @ self.x + self.B @ u + noise + disturbance
        return self.x.copy()
    
    def reset(self, x0):
        self.x = x0.copy()

class APIInterface(SystemInterface):
    def __init__(self, system_config, mpc_config):
        self.system_config = system_config
        self.mpc_config = mpc_config
        self.endpoint = system_config.api_config['endpoint']
        self.timeout = system_config.api_config['timeout']
        self.headers = system_config.api_config['headers']
    
    def get_state(self):
        try:
            response = requests.get(
                f"{self.endpoint}/state",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return np.array(response.json()['state'])
        except requests.exceptions.RequestException as e:
            print(f"Error getting state: {e}")
            return None
    
    def apply_input(self, u):
        try:
            response = requests.post(
                f"{self.endpoint}/input",
                headers=self.headers,
                json={'input': u.tolist()},
                timeout=self.timeout
            )
            response.raise_for_status()
            return np.array(response.json()['state'])
        except requests.exceptions.RequestException as e:
            print(f"Error applying input: {e}")
            return None
    
    def reset(self, x0):
        try:
            response = requests.post(
                f"{self.endpoint}/reset",
                headers=self.headers,
                json={'state': x0.tolist()},
                timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error resetting system: {e}")

def create_system_interface(system_config, mpc_config):
    """Factory function to create appropriate system interface"""
    if system_config.system_type == 'simulation':
        return SimulationInterface(system_config, mpc_config)
    elif system_config.system_type == 'api':
        return APIInterface(system_config, mpc_config)
    else:
        raise ValueError(f"Unknown system type: {system_config.system_type}") 