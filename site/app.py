from flask import Flask, render_template, jsonify, request
import numpy as np
from config import MPCConfig, SystemConfig
from mpc_controller import MPCController
import json

app = Flask(__name__)

# Global variables to store configurations and results
mpc_config = MPCConfig()
system_config = SystemConfig()
controller = None
last_results = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    config = {
        'mpc': {
            'dt': mpc_config.dt,
            'N': mpc_config.N,
            'Q': mpc_config.Q.tolist(),
            'R': mpc_config.R.tolist(),
            'x_min': mpc_config.x_min.tolist(),
            'x_max': mpc_config.x_max.tolist(),
            'u_min': mpc_config.u_min.tolist(),
            'u_max': mpc_config.u_max.tolist(),
            'terminal_region_radius': mpc_config.terminal_region_radius,
            'simulation_time': mpc_config.simulation_time,
            'initial_state': mpc_config.initial_state.tolist(),
            'reference_state': mpc_config.reference_state.tolist()
        },
        'system': {
            'system_type': system_config.system_type,
            'api_config': system_config.api_config,
            'sim_config': system_config.sim_config
        }
    }
    return jsonify(config)

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration"""
    global mpc_config, system_config, controller
    
    data = request.json
    
    # Update MPC config
    mpc_config.dt = float(data['mpc']['dt'])
    mpc_config.N = int(data['mpc']['N'])
    mpc_config.Q = np.array(data['mpc']['Q'])
    mpc_config.R = np.array(data['mpc']['R'])
    mpc_config.x_min = np.array(data['mpc']['x_min'])
    mpc_config.x_max = np.array(data['mpc']['x_max'])
    mpc_config.u_min = np.array(data['mpc']['u_min'])
    mpc_config.u_max = np.array(data['mpc']['u_max'])
    mpc_config.terminal_region_radius = float(data['mpc']['terminal_region_radius'])
    mpc_config.simulation_time = float(data['mpc']['simulation_time'])
    mpc_config.initial_state = np.array(data['mpc']['initial_state'])
    mpc_config.reference_state = np.array(data['mpc']['reference_state'])
    
    # Update system config
    system_config.system_type = data['system']['system_type']
    system_config.api_config = data['system']['api_config']
    system_config.sim_config = data['system']['sim_config']
    
    # Recreate controller with new config
    controller = MPCController(mpc_config, system_config)
    
    return jsonify({'status': 'success'})

@app.route('/api/run', methods=['POST'])
def run_simulation():
    """Run MPC simulation"""
    global controller, last_results
    
    if controller is None:
        controller = MPCController(mpc_config, system_config)

    t, x, u, mode = controller.run()

    # Store results
    last_results = {
        't': t.tolist(),
        'x': x.tolist(),
        'u': u.tolist(),
        'mode': mode.tolist()
    }

    return jsonify(last_results)

@app.route('/api/results', methods=['GET'])
def get_results():
    """Get last simulation results"""
    if last_results is None:
        return jsonify({'error': 'No results available'})
    return jsonify(last_results)

if __name__ == '__main__':
    app.run(debug=True) 