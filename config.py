import numpy as np

class MPCConfig:
    def __init__(self):
        # System parameters
        self.dt = 1  # Time step
        self.N = 8    # Prediction horizon (increased for better trajectory planning)
        
        # State vector: [x, vx, y, vy, z, vz, phi, theta, psi]
        # x,y,z: position
        # vx,vy,vz: linear velocities 
        # phi,theta,psi: roll, pitch, yaw angles
        
        # System dynamics (linearized quadrotor)
        g = 9.81  # gravity
        self.A = np.array([
            [1, self.dt, 0, 0, 0, 0, 0, 0, 0],     # x
            [0, 1, 0, 0, 0, 0, 0, g, 0],           # vx (affected by pitch)
            [0, 0, 1, self.dt, 0, 0, 0, 0, 0],     # y
            [0, 0, 0, 1, 0, 0, -g, 0, 0],          # vy (affected by roll)
            [0, 0, 0, 0, 1, self.dt, 0, 0, 0],     # z
            [0, 0, 0, 0, 0, 1, 0, 0, -g],          # vz (affected by gravity)
            [0, 0, 0, 0, 0, 0, 1, 0, 0],           # phi
            [0, 0, 0, 0, 0, 0, 0, 1, 0],           # theta
            [0, 0, 0, 0, 0, 0, 0, 0, 1]            # psi
        ])

        # Input vector: [thrust, roll_rate, pitch_rate, yaw_rate]
        self.B = np.array([
            [0, 0, 0, 0],           # x (affected by pitch/roll through dynamics)
            [0, 0, 0, 0],           # vx
            [0, 0, 0, 0],           # y (affected by pitch/roll through dynamics)
            [0, 0, 0, 0],           # vy
            [self.dt**2/2, 0, 0, 0], # z (directly affected by thrust)
            [self.dt, 0, 0, 0],      # vz
            [0, self.dt, 0, 0],      # phi (roll)
            [0, 0, self.dt, 0],      # theta (pitch)
            [0, 0, 0, self.dt]       # psi (yaw)
        ])


        # Cost matrices
        self.Q = np.diag([1.0, # x
                          0.1, # vx
                          1.0, # y
                          0.1, # vy
                          10, # z
                          0.1, # vz
                          0.01, # phi
                          0.01, # theta
                          0.1])  # psi
        
        self.R = np.diag([0.001, # thrust
                          0.1, # roll_rate
                          0.1, # pitch_rate
                          0.01])  # yaw_rate
        
        # System dimensions
        self.nx = self.A.shape[0]    # state dimension (9)
        self.nu = self.B.shape[1]    # input dimension (4)
        
        # Constraints
        self.x_min = np.array([-20, -5, -20, -5, -20, -5, -np.pi/2, -np.pi/2, -np.pi]) # More relaxed
        self.x_max = np.array([20, 5, 20, 5, 20, 5, np.pi/2, np.pi/2, np.pi])
        self.u_min = np.array([0, -5, -5, -5])     # min thrust and angular rates
        self.u_max = np.array([100, 5, 5, 5])      # max thrust and angular rates
        
        # Terminal region
        self.terminal_region_radius = 1
        
        # API settings
        self.use_api = False
        self.api_endpoint = "http://localhost:8000"
        self.api_timeout = 1.0  # seconds
        
        # Simulation settings
        self.simulation_time = 8
        self.initial_state = np.zeros(self.nx)  # Start at origin with zero velocities and angles
        self.initial_input = np.zeros(self.nu)
        self.reference_state = np.array([10.0, 0.0, 10.0, 0.0, 10, 0.0, 0.0, 0.0, 0.0])  # Target position [1,1,1]
        
        # Test dimensions
        self.test()

    def test(self):
        """Test system configuration for dimensional consistency and control properties.
        
        Raises:
            ValueError: If matrix dimensions are inconsistent or system properties are invalid
        """
        
        # Test dimensions of A and Q matrices
        if self.A.shape[0] != self.Q.shape[0] or self.A.shape[1] != self.Q.shape[1]:
            raise ValueError(f"A matrix shape {self.A.shape} does not match Q matrix shape {self.Q.shape}")
            
        # Test dimensions of B and R matrices 
        if self.B.shape[1] != self.R.shape[0]:
            raise ValueError(f"B matrix columns {self.B.shape[1]} does not match R matrix rows {self.R.shape[0]}")

        # Test system observability
        obs_matrix = np.vstack([self.A**i for i in range(self.nx)])
        if np.linalg.matrix_rank(obs_matrix) != self.nx:
            raise ValueError(f"System is not observable. Rank of observability matrix: {np.linalg.matrix_rank(obs_matrix)}, should be {self.nx}")
            
        # Test system reachability
        reach_matrix = np.hstack([np.linalg.matrix_power(self.A, i) @ self.B for i in range(self.nx)])
        if np.linalg.matrix_rank(reach_matrix) != self.nx:
            print(f"System is not reachable. Rank of reachability matrix: {np.linalg.matrix_rank(reach_matrix)}, should be {self.nx}")
            print(f"Testing controllability...")
    
            # Test system controllability
            ctrl_matrix = np.hstack([np.linalg.matrix_power(self.A, i) @ self.B for i in range(self.nx)])
            if np.linalg.matrix_rank(ctrl_matrix) != self.nx:
                raise ValueError(f"System is not controllable. Rank of controllability matrix: {np.linalg.matrix_rank(ctrl_matrix)}, should be {self.nx}")
        
        # Test dimensions of x_min and x_max matrices
        if len(self.initial_state) != self.A.shape[0]:
            raise ValueError(f"Initial state length {len(self.initial_state)} does not match system dimension {self.A.shape[0]}")
            
        if len(self.reference_state) != self.A.shape[0]:
            raise ValueError(f"Reference state length {len(self.reference_state)} does not match system dimension {self.A.shape[0]}")
        
        # Test dimensions of constraint vectors
        if len(self.x_min) != self.nx or len(self.x_max) != self.nx:
            raise ValueError(f"State constraint vectors length {len(self.x_min)}, {len(self.x_max)} do not match state dimension {self.nx}")
            
        if len(self.u_min) != self.nu or len(self.u_max) != self.nu:
            raise ValueError(f"Input constraint vectors length {len(self.u_min)}, {len(self.u_max)} do not match input dimension {self.nu}")
        


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