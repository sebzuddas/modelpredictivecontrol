import numpy as np

class MPCConfig:
    def __init__(self):
        # System parameters
        self.dt = 0.1  # time step
        self.N = 10    # prediction horizon
        
        # System dynamics 
        self.A = np.array([[1, self.dt],
                          [0, 1]])
        self.B = np.array([[0.5 * self.dt**2],
                          [self.dt]])
        
        # System dimensions based on A and B matrices
        self.nx = self.A.shape[0]    # state dimension (from A matrix)
        self.nu = self.B.shape[1]    # input dimension (from B matrix)
        
        # Cost matrices
        self.Q = np.diag([1.0, 0.1])  # state cost
        self.R = np.array([[0.1]])    # input cost
        
        # Constraints
        self.x_min = np.array([-np.inf, -10.0])  # state constraints
        self.x_max = np.array([np.inf, 10.0])
        self.u_min = np.array([-10.0])           # input constraints
        self.u_max = np.array([10.0])
        
        # Terminal region
        self.terminal_region_radius = 0.1
        
        # API settings
        self.use_api = False
        self.api_endpoint = "http://localhost:8000"
        self.api_timeout = 1.0  # seconds
        
        # Simulation settings
        self.simulation_time = 10.0
        self.initial_state = np.array([0.0, 0.0])
        self.reference_state = np.array([1.0, 0.0])
        
        # Test dimensions
        self.test()   

    def test(self):
        
        # Test dimensions of A and Q matrices
        if self.A.shape[0] != self.Q.shape[0] or self.A.shape[1] != self.Q.shape[1]:
            print(f"Error: A matrix shape {self.A.shape} does not match Q matrix shape {self.Q.shape}")
            return False
            
        # Test dimensions of B and R matrices 
        if self.B.shape[1] != self.R.shape[0]:
            print(f"Error: B matrix columns {self.B.shape[1]} does not match R matrix rows {self.R.shape[0]}")
            return False
        
        # Test dimensions of x_min and x_max matrices
        if len(self.initial_state) != self.A.shape[0]:
            print(f"Error: initial state length {len(self.initial_state)} does not match system dimension {self.A.shape[0]}")
            return False
            
        if len(self.reference_state) != self.A.shape[0]:
            print(f"Error: reference state length {len(self.reference_state)} does not match system dimension {self.A.shape[0]}")
            return False
        
        # Test dimensions of constraint vectors
        if len(self.x_min) != self.nx or len(self.x_max) != self.nx:
            print(f"Error: state constraint vectors length {len(self.x_min)}, {len(self.x_max)} do not match state dimension {self.nx}")
            return False
            
        if len(self.u_min) != self.nu or len(self.u_max) != self.nu:
            print(f"Error: input constraint vectors length {len(self.u_min)}, {len(self.u_max)} do not match input dimension {self.nu}")
            return False

        print("All tests passed!")
        return True

class SystemConfig:
    def __init__(self):
        # System type: 'simulation' or 'api'
        self.system_type = 'simulation'
        
        # API configuration
        self.api_config = {
            'endpoint': 'http://localhost:8000',
            'timeout': 1.0,
            'headers': {
                'Content-Type': 'application/json'
            }
        }
        
        # Simulation configuration
        self.sim_config = {
            'dt': 0.1,
            'noise_level': 0.0,  # Add noise to simulation
            'disturbance_level': 0.0  # Add disturbances to simulation
        } 