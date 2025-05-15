import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from config import MPCConfig, SystemConfig
from system_interface import create_system_interface
import os
import json
from datetime import datetime
from tqdm import tqdm


class MPCController:
    def __init__(self, mpc_config, system_config):
        """
        Initialize dual-mode MPC controller
        
        Parameters:
        -----------
        mpc_config : MPCConfig
            MPC configuration object
        system_config : SystemConfig
            System configuration object
        """
        self.config = mpc_config
        self.system = create_system_interface(system_config, mpc_config)
        
        # System dimensions
        self.nx = mpc_config.nx
        self.nu = mpc_config.nu
        
        # System dynamics
        self.x = ca.SX.sym('x', self.nx)
        self.u = ca.SX.sym('u', self.nu)
        
        # Define system dynamics
        self.f = ca.mtimes(self.config.A, self.x) + ca.mtimes(self.config.B, self.u)
        
        # Create integrator
        self.dae = {'x': self.x, 'p': self.u, 'ode': self.f}
        # Try rk4 integrator with more options
        self.integrator = ca.integrator('F', 'rk',
                                      self.dae,
                                      0.0,  # t0
                                      self.config.dt,  # tf
                                      {
                                          'number_of_finite_elements': 4,
                                          'simplify': True,
                                          'expand': True
                                      })
        
        # Compute LQR gain for terminal mode
        self.compute_lqr_gain()
        
        # MPC setup
        self.setup_mpc()
    
    def compute_lqr_gain(self):
        """Compute LQR gain for terminal mode"""
        try:
            # Solve discrete-time LQR for terminal cost
            P = linalg.solve_discrete_are(self.config.A, self.config.B, 
                                        self.config.Q, self.config.R)
            self.K = -np.linalg.inv(self.config.R + self.config.B.T @ P @ self.config.B) @ self.config.B.T @ P @ self.config.A
            self.P = P
            print(f"LQR gain computed successfully!")# {self.K}")
        except np.linalg.LinAlgError as e:
            print(f"DARE failed: {str(e)}")
            raise  # Don't fall back to zero gain, this is critical
    
    def setup_mpc(self):
        """Setup the dual-mode MPC optimization problem"""
        self.opti = ca.Opti()
        
        # Decision variables
        self.X = self.opti.variable(self.nx, self.config.N + 1)  # States from 0 to N
        self.U = self.opti.variable(self.nu, self.config.N)      # Inputs from 0 to N-1
        
        # Parameters
        self.x0 = self.opti.parameter(self.nx)
        self.x_ref = self.opti.parameter(self.nx)
        
        # Cost function
        cost = 0
        # Stage cost (0 to N-1)
        for k in range(self.config.N):
            state_error = self.X[:, k] - self.x_ref
            cost += ca.mtimes([state_error.T, self.config.Q, state_error])
            cost += ca.mtimes([self.U[:, k].T, self.config.R, self.U[:, k]])
        
        # Terminal cost (N to ∞)
        terminal_error = self.X[:, -1] - self.x_ref
        cost += ca.mtimes([terminal_error.T, self.P, terminal_error])
        
        self.opti.minimize(cost)
        
        # Constraints
        for k in range(self.config.N):
            # System dynamics
            self.opti.subject_to(
                self.X[:, k+1] == self.integrator(x0=self.X[:, k], p=self.U[:, k])['xf']
            )
            
            # State constraints
            self.opti.subject_to(
                self.opti.bounded(self.config.x_min, self.X[:, k], self.config.x_max)
            )
            
            # Input constraints
            self.opti.subject_to(
                self.opti.bounded(self.config.u_min, self.U[:, k], self.config.u_max)
            )
        
        # Terminal constraint (ensure LQR is valid)
        self.opti.subject_to(
            ca.mtimes([terminal_error.T, self.P, terminal_error]) <= self.config.terminal_region_radius**2
        )
        
        # Initial state constraint
        self.opti.subject_to(self.X[:, 0] == self.x0)
        
        # Solver
        self.opti.solver('ipopt', {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 1000,
            'ipopt.tol': 1e-3
        })
    
    def solve(self, x0, x_ref):
        """Solve the MPC problem"""
        try:
            self.opti.set_value(self.x0, x0)
            self.opti.set_value(self.x_ref, x_ref)
            sol = self.opti.solve()
            u = sol.value(self.U[:, 0])  # Only use first input
            # print(f"MPC solution found: {u}")
            # print(f"Current state: \n{x0}")
            # print(f"Reference state: \n{x_ref}")
            # print(f"Difference: \n{x_ref - x0}")
            return u
        except Exception as e:
            # print(f"MPC solve FAILED: {str(e)}")
            # print(f"Current state: \n{x0}")
            # print(f"Reference state: \n{x_ref}")
            # print(f"Difference: \n{x_ref - x0}")
            # Use LQR as fallback
            return self.get_terminal_control(x0, x_ref)
    
    def get_terminal_control(self, x, x_ref):
        """Get control input from LQR controller"""
        x_np = np.array(x).flatten()
        x_ref_np = np.array(x_ref).flatten()
        
        u = (self.K @ (x_np - x_ref_np)).flatten()
        return np.clip(u, self.config.u_min, self.config.u_max)
    
    def run(self, x0=None, u0=None, x_ref=None, T=None):
        """Run the MPC controller"""
        if x0 is None:
            x0 = self.config.initial_state
        if u0 is None:
            u0 = self.config.initial_input
        if x_ref is None:
            x_ref = self.config.reference_state
        if T is None:
            T = self.config.simulation_time
        
        # Reset system
        self.system.reset(x0)
        
        # Simulation setup
        N_sim = int(T / self.config.dt)
        t = np.linspace(0, T, N_sim + 1)
        x = np.zeros((self.nx, N_sim + 1))
        u = np.zeros((self.nu, N_sim))
        mode = np.zeros(N_sim)  # 0 for MPC, 1 for LQR
        
        # Initial state and input
        x[:, 0] = x0
        u[:, 0] = u0  # Set initial input

        self.print_simulation_parameters()

        next_state = self.system.apply_input(u[:, 0])
        if next_state is None:
            print("Error: Could not apply initial control input")
            return t, x, u, mode
        x[:, 1] = next_state
        
        # Control loop (start from k=1 since we've already applied initial input)

        for k in tqdm(range(1, N_sim), desc="Simulating"):
            # print("Simulation_iteration: ", {k})
            current_state = self.system.get_state()
            if current_state is None:
                # print("Error: Could not get system state") 
                break
            
            try:
                # Try MPC first
                u[:, k] = self.solve(current_state, x_ref)
                mode[k] = 0
            except:
                # Fall back to LQR if MPC fails
                u[:, k] = self.get_terminal_control(current_state, x_ref)
                mode[k] = 1
                # print(f"Switching to LQR at step {k}")
            
            # Apply control input
            next_state = self.system.apply_input(u[:, k])
            if next_state is None:
                # print("Error: Could not apply control input")
                break
            
            # Store results
            x[:, k+1] = next_state
        
        return t, x, u, mode

    def print_simulation_parameters(self):

        # Apply initial input first
        print("*"*50)
        print("Simulation Configuration")
        print("*"*50)
        print("Starting parameters:")
        print(f"Initial state: \n{self.config.initial_state}")
        print(f"Initial reference state: \n{self.config.reference_state}")
        print(f"Initial input: \n{self.config.initial_input}")
        print("-"*50)
        print("Tuning Parameters:")
        print(f"Q: \n{self.config.Q}")
        print(f"R: \n{self.config.R}")
        print(f"N: {self.config.N}")
        print(f"dt: {self.config.dt}")
        print(f"terminal region radius: {self.config.terminal_region_radius}")
        print("-"*50)
        print("System dynamics:")
        print(f"A: \n{self.config.A}")
        print(f"B: \n{self.config.B}")
        print("-"*50)
        print("State & Input constraints:")
        print(f"x_min: \n{self.config.x_min}")
        print(f"x_max: \n{self.config.x_max}")
        print(f"u_min: \n{self.config.u_min}")
        print(f"u_max: \n{self.config.u_max}")
        print("-"*50)
        print("*"*50)


def plot_results(t, x, u, x_ref, mode, config):
    """Plot simulation results"""
    # Calculate number of states and inputs
    n_states = x.shape[0]  # Number of states
    n_inputs = u.shape[0]  # Number of inputs
    

    # Plot all states on one axis
    plt.subplot(3, 1, 1)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_states))
    for i in range(n_states):
        plt.plot(t, x[i, :], color=colors[i], label=f'State {i+1}')
        plt.plot(t, np.ones_like(t) * x_ref[i], '--', color=colors[i]) # label=f'Reference {i+1}'
        plt.axhline(y=config.x_max[i], color=colors[i], linestyle=':') # label=f'State {i+1} limits'
        plt.axhline(y=config.x_min[i], color=colors[i], linestyle=':')
    
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('States')
    
    # Plot all inputs on one axis
    plt.subplot(3, 1, 2)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_inputs))
    for i in range(n_inputs):
        plt.plot(t[:-1], u[i, :], color=colors[i], label=f'Input {i+1}') # , label=f'Input {i+1}'
        plt.axhline(y=config.u_max[i], color=colors[i], linestyle=':') # , label=f'Input {i+1} limits'
        plt.axhline(y=config.u_min[i], color=colors[i], linestyle=':')
    
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Inputs')
    
    # 3D position plot
    ax1 = plt.subplot(3, 1, 3, projection='3d')
    ax1.plot(x[0,:], x[2,:], x[4,:], 'b-', label='Trajectory')
    ax1.scatter(x_ref[0], x_ref[2], x_ref[4], color='r', marker='*', s=100, label='Target')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_zlabel('Z [m]')
    ax1.grid(True)
    ax1.legend()
    
    
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    # Create configurations
    mpc_config = MPCConfig()
    system_config = SystemConfig()
    
    # Create and run controller
    controller = MPCController(mpc_config, system_config)
    t, x, u, mode = controller.run()
    
    # Plot results
    plot_results(t, x, u, mpc_config.reference_state, mode, mpc_config) 