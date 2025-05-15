from config import MPCConfig, SystemConfig
from mpc_controller import MPCController, plot_results
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

def run_simulation():
    """Run the MPC controller in simulation mode"""
    print("Running MPC in simulation mode...")
    
    # Create configurations
    mpc_config = MPCConfig()
    system_config = SystemConfig()

    
    # # Configure simulation parameters
    # mpc_config.simulation_time = 10.0
    # mpc_config.initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # mpc_config.reference_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Add some noise to the simulation
    system_config.sim_config['noise_level'] = 0.01
    system_config.sim_config['disturbance_level'] = 0.001
    
    # Create and run controller
    controller = MPCController(mpc_config, system_config)
    t, x, u, mode = controller.run()
    
    
    # Save data
    save_simulation_data(t, x, u, mode, mpc_config)
    # Plot results
    plot_results(t, x, u, mpc_config.reference_state, mode, mpc_config)


def save_simulation_data(t, x, u, mode, config, timestamp=None):
    """Save simulation data to a structured format"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join('data', timestamp)
    os.makedirs(data_dir, exist_ok=True)
    
    # Save states and inputs as CSV
    states_df = pd.DataFrame(x.T, columns=['x', 'vx', 'y', 'vy', 'z', 'vz', 'phi', 'theta', 'psi'])
    inputs_df = pd.DataFrame(u.T, columns=['thrust', 'roll_rate', 'pitch_rate', 'yaw_rate'])
    time_df = pd.DataFrame(t, columns=['time'])
    
    states_df.to_csv(os.path.join(data_dir, 'states.csv'), index=False)
    inputs_df.to_csv(os.path.join(data_dir, 'inputs.csv'), index=False)
    time_df.to_csv(os.path.join(data_dir, 'time.csv'), index=False)
    
    # Save configuration parameters
    config_dict = {
        'dt': float(config.dt),  # Convert numpy types to native Python types
        'N': int(config.N),
        'Q': config.Q.tolist(),
        'R': config.R.tolist(),
        'terminal_region_radius': float(config.terminal_region_radius),
        'simulation_time': float(config.simulation_time),
        'initial_state': config.initial_state.tolist(),
        'reference_state': config.reference_state.tolist(),
        'initial_input': config.initial_input.tolist(),
        'state_minima': config.x_min.tolist(),
        'state_maxima': config.x_max.tolist(),
        'input_minima': config.u_min.tolist(),
        'input_maxima': config.u_max.tolist()
    }
    
    try:
        with open(os.path.join(data_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)
    except Exception as e:
        print(f"Error saving config.json: {e}")
        print(f"Config dict: {config_dict}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'simulation_duration': float(t[-1]),  # Convert numpy float to native Python float
        'final_error': float(np.linalg.norm(x[:, -1] - config.reference_state)),
        'mpc_usage': float(np.mean(mode == 0)),  # Convert numpy types to native Python types
        'lqr_usage': float(np.mean(mode == 1))
    }
    
    try:
        with open(os.path.join(data_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
    except Exception as e:
        print(f"Error saving metadata.json: {e}")
        print(f"Metadata: {metadata}")
    
    return data_dir

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