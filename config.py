import numpy as np

class MPCConfig:
    def __init__(self):
        # System parameters
        self.dt = 0.1  # time step
        self.N = 5    # prediction horizon
        
        # System dynamics 
        self.A = np.array([[1, self.dt, 0, 0, 0, 0, 0, 1, 0],
                        [0, 1, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 1, self.dt, 0, 1, 0, 0, 1],
                        [0, 1, 0, 1, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1, self.dt, 0, 0, 0],
                        [0, 1, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, self.dt, 0],
                        [0, 1, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 1]])
        
        # # Square of A matrix for testing
        # self.A2 = np.matmul(self.A, self.A)
        # print(self.A2)
        # exit()

        self.B = np.array([[0.5 * self.dt**2, 0, 0],
                           [self.dt, 0, 0],
                           [0, 0.5 * self.dt**2, 0],
                           [0, self.dt, 0],
                           [1, 0, 0.5 * self.dt**2],
                           [1, 0, self.dt],
                           [1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

        # Cost matrices - make Q more positive definite
        self.Q = np.diag([1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # state cost
        self.R = np.diag([1.0, 1.0, 1.0])    # input cost
        
        # System dimensions based on A and B matrices
        self.nx = self.A.shape[0]    # state dimension (from A matrix)
        self.nu = self.B.shape[1]    # input dimension (from B matrix)
        
        
        # Constraints
        self.x_min = np.array([-10, -10, -10, -10, -10, -10, -10, -10, -10])  # state constraints
        self.x_max = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10]) # Pass valid JSON into this
        self.u_min = np.array([-10, -10, -10])           # input constraints
        self.u_max = np.array([10, 10, 10])
        
        # Terminal region
        self.terminal_region_radius = 0.1
        
        # API settings
        self.use_api = False
        self.api_endpoint = "http://localhost:8000"
        self.api_timeout = 1.0  # seconds
        
        # Simulation settings
        self.simulation_time = 10.0
        self.initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.reference_state = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        
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
        # Test system observability
        obs_matrix = np.vstack([self.A**i for i in range(self.nx)])
        if np.linalg.matrix_rank(obs_matrix) != self.nx:
            print(f"Warning: System is not observable. Rank of observability matrix: {np.linalg.matrix_rank(obs_matrix)}, should be {self.nx}")
            
        # Test system reachability
        reach_matrix = np.hstack([np.linalg.matrix_power(self.A, i) @ self.B for i in range(self.nx)])
        if np.linalg.matrix_rank(reach_matrix) != self.nx:
            print(f"Warning: System is not reachable. Rank of reachability matrix: {np.linalg.matrix_rank(reach_matrix)}, should be {self.nx}")
            
            # Test system controllability
            ctrl_matrix = np.hstack([np.linalg.matrix_power(self.A, i) @ self.B for i in range(self.nx)])
            if np.linalg.matrix_rank(ctrl_matrix) != self.nx:
                print(f"Warning: System is not controllable. Rank of controllability matrix: {np.linalg.matrix_rank(ctrl_matrix)}, should be {self.nx}")
        
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
        


        print("Testing Finished!")
        
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