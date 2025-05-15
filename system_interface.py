import numpy as np
import requests
import time
from abc import ABC, abstractmethod
from pysindy import SINDy
from pysindy.optimizers import STLSQ
from pysindy.feature_library import PolynomialLibrary
from config import MPCConfig
# from sindy_identifier import identify_and_update_system

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

class SINDYcIdentifier:
    def __init__(self, dt=0.05, threshold=0.1, max_iter=20):
        """
        Initialize SINDYc identifier
        
        Parameters:
        -----------
        dt : float
            Time step for system identification
        threshold : float
            Threshold for sparse regression
        max_iter : int
            Maximum number of iterations for sparse regression
        """
        self.dt = dt
        self.threshold = threshold
        self.max_iter = max_iter
        
        # Initialize SINDy model
        self.optimizer = STLSQ(threshold=threshold, max_iter=max_iter)
        self.feature_library = PolynomialLibrary(degree=2)
        self.model = SINDy(
            optimizer=self.optimizer,
            feature_library=self.feature_library,
            discrete_time=True
        )
        
        # Store identified system matrices
        self.A = None
        self.B = None
        
    def collect_data(self, system_interface, T=10.0, noise_level=0.01):
        """
        Collect training data from system
        
        Parameters:
        -----------
        system_interface : SystemInterface
            Interface to the system
        T : float
            Duration of data collection
        noise_level : float
            Level of noise to add to measurements
        """
        # Reset system
        x0 = np.zeros(9)  # Assuming 9 states
        system_interface.reset(x0)
        
        # Setup data collection
        N = int(T / self.dt)
        x = np.zeros((9, N + 1))  # States
        u = np.zeros((4, N))      # Inputs
        
        # Collect data
        x[:, 0] = x0
        for k in range(N):
            # Apply random input
            u[:, k] = np.random.uniform(-1, 1, 4)
            next_state = system_interface.apply_input(u[:, k])
            if next_state is None:
                raise ValueError("Failed to collect system data")
            x[:, k + 1] = next_state + np.random.normal(0, noise_level, 9)
        
        return x, u
    
    def identify_system(self, x, u):
        """
        Identify system dynamics using SINDYc
        
        Parameters:
        -----------
        x : np.ndarray
            State trajectory data
        u : np.ndarray
            Input trajectory data
        """
        # Fit model
        self.model.fit(x.T, u=u.T, t=self.dt)
        
        # Extract linear system matrices
        self.A, self.B = self._extract_linear_system()
        
        return self.A, self.B
    
    def _extract_linear_system(self):
        """
        Extract linear system matrices from SINDy model
        """
        # Get model coefficients
        coeffs = self.model.coefficients()
        
        # Extract A and B matrices
        nx = 9  # Number of states
        nu = 4  # Number of inputs
        
        A = np.zeros((nx, nx))
        B = np.zeros((nx, nu))
        
        # Extract linear terms
        for i in range(nx):
            # State terms
            for j in range(nx):
                state_term = f'x{j}'
                if state_term in self.model.feature_names:
                    idx = self.model.feature_names.index(state_term)
                    A[i, j] = coeffs[i, idx]
            
            # Input terms
            for j in range(nu):
                input_term = f'u{j}'
                if input_term in self.model.feature_names:
                    idx = self.model.feature_names.index(input_term)
                    B[i, j] = coeffs[i, idx]
        
        return A, B
    
    def update_mpc_config(self, mpc_config):
        """
        Update MPC configuration with identified system
        
        Parameters:
        -----------
        mpc_config : MPCConfig
            MPC configuration to update
        """
        if self.A is None or self.B is None:
            raise ValueError("System not identified yet")
        
        mpc_config.A = self.A
        mpc_config.B = self.B
        mpc_config.test()  # Test updated system
        
        return mpc_config

def create_system_interface(system_config, mpc_config):
    """Factory function to create appropriate system interface"""
    if system_config.system_type == 'simulation':
        return SimulationInterface(system_config, mpc_config)
    elif system_config.system_type == 'api':
        return APIInterface(system_config, mpc_config)
    else:
        raise ValueError(f"Unknown system type: {system_config.system_type}")

# def identify_and_update_system(system_interface, mpc_config, T=10.0):
#     """
#     Identify system and update MPC configuration
    
#     Parameters:
#     -----------
#     system_interface : SystemInterface
#         Interface to the system
#     mpc_config : MPCConfig
#         MPC configuration to update
#     T : float
#         Duration of data collection
#     """
#     # Create identifier
#     identifier = SINDYcIdentifier(dt=mpc_config.dt)
    
#     # Collect data
#     x, u = identifier.collect_data(system_interface, T=T)
    
#     # Identify system
#     A, B = identifier.identify_system(x, u)
    
#     # Update MPC config
#     mpc_config = identifier.update_mpc_config(mpc_config)
    
#     return mpc_config, A, B 

def run_simulation():
    """Run the MPC controller in simulation mode"""
    print("Running MPC in simulation mode...")
    
    # Create configurations
    mpc_config = MPCConfig()
    system_config = SystemConfig()
    
    # Create system interface
    system = create_system_interface(system_config, mpc_config)
    
    # Identify system and update config
    print("Identifying system...")
    mpc_config, A, B = identify_and_update_system(system, mpc_config)
    print("System identified:")
    print("A matrix:")
    print(A)
    print("B matrix:")
    print(B)
    
    # Create and run controller
    controller = MPCController(mpc_config, system_config)
    t, x, u, mode = controller.run()
    
    # Plot results
    plot_results(t, x, u, mpc_config.reference_state, mode, mpc_config) 