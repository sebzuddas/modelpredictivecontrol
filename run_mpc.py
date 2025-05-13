from config import MPCConfig, SystemConfig
from mpc_controller import MPCController, plot_results

def run_simulation():
    """Run the MPC controller in simulation mode"""
    print("Running MPC in simulation mode...")
    
    # Create configurations
    mpc_config = MPCConfig()
    system_config = SystemConfig()
    
    # Configure simulation parameters
    mpc_config.simulation_time = 10.0
    mpc_config.initial_state = np.array([0.0, 0.0])
    mpc_config.reference_state = np.array([1.0, 0.0])
    
    # Add some noise to the simulation
    system_config.sim_config['noise_level'] = 0.01
    system_config.sim_config['disturbance_level'] = 0.005
    
    # Create and run controller
    controller = MPCController(mpc_config, system_config)
    t, x, u, mode = controller.run()
    
    # Plot results
    plot_results(t, x, u, mpc_config.reference_state, mode, mpc_config)

def run_api():
    """Run the MPC controller connected to an API"""
    print("Running MPC in API mode...")
    
    # Create configurations
    mpc_config = MPCConfig()
    system_config = SystemConfig()
    
    # Configure API settings
    system_config.system_type = 'api'
    system_config.api_config['endpoint'] = 'http://localhost:8000'  # Change this to your API endpoint
    system_config.api_config['timeout'] = 1.0
    
    # Create and run controller
    controller = MPCController(mpc_config, system_config)
    t, x, u, mode = controller.run()
    
    # Plot results
    plot_results(t, x, u, mpc_config.reference_state, mode, mpc_config)

if __name__ == "__main__":
    import numpy as np
    
    # Choose mode: 'simulation' or 'api'
    mode = 'simulation'
    
    if mode == 'simulation':
        run_simulation()
    elif mode == 'api':
        run_api()
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: 'simulation', 'api'") 