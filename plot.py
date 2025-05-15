import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import os
import argparse
from datetime import datetime

def load_simulation_data(simulation_dir):
    """Load simulation data from a directory"""
    # Find the iteration subdirectory
    iteration_dirs = [d for d in os.listdir(simulation_dir) 
                     if os.path.isdir(os.path.join(simulation_dir, d)) and d.startswith('iteration_')]
    
    if not iteration_dirs:
        raise FileNotFoundError(f"No iteration directories found in {simulation_dir}")
    
    # Use the last iteration
    latest_iteration = sorted(iteration_dirs)[-1]
    iteration_path = os.path.join(simulation_dir, latest_iteration)
    
    print(f"Loading data from iteration: {latest_iteration}")
    
    states = pd.read_csv(os.path.join(iteration_path, 'states.csv'))
    inputs = pd.read_csv(os.path.join(iteration_path, 'inputs.csv'))
    time = pd.read_csv(os.path.join(iteration_path, 'time.csv'))
    config = json.load(open(os.path.join(iteration_path, 'config.json')))
    
    # Print column names for debugging
    print("Time columns:", time.columns.tolist())
    
    return states, inputs, time, config

def plot_3d_trajectory(states, title="Quadrotor Trajectory"):
    """Plot 3D trajectory using plotly"""
    fig = go.Figure(data=[go.Scatter3d(
        x=states['x'],
        y=states['y'],
        z=states['z'],
        mode='lines+markers',
        marker=dict(
            size=2,
            color=states.index,  # Color by time
            colorscale='Viridis',
        ),
        line=dict(
            color='darkblue',
            width=2
        )
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Z Position',
            aspectmode='data'
        ),
        showlegend=False
    )
    
    return fig

def plot_states(states, time, title="State Evolution"):
    """Plot all states over time"""
    fig = go.Figure()
    
    # Get time column name (it might be 'time' or 't')
    time_col = time.columns[0]  # Use first column as time
    
    for col in states.columns:
        fig.add_trace(go.Scatter(
            x=time[time_col],
            y=states[col],
            name=col,
            mode='lines'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='State Value',
        showlegend=True
    )
    
    return fig

def plot_inputs(inputs, time, title="Control Inputs"):
    """Plot control inputs over time"""
    fig = go.Figure()
    
    # Get time column name (it might be 'time' or 't')
    time_col = time.columns[0]  # Use first column as time
    
    for col in inputs.columns:
        fig.add_trace(go.Scatter(
            x=time[time_col],
            y=inputs[col],
            name=col,
            mode='lines'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Input Value',
        showlegend=True
    )
    
    return fig

def plot_simulation(simulation_dir):
    """Plot all aspects of a simulation"""
    try:
        states, inputs, time, config = load_simulation_data(simulation_dir)
        
        # Create figures
        traj_fig = plot_3d_trajectory(states)
        states_fig = plot_states(states, time)
        inputs_fig = plot_inputs(inputs, time)
        
        # Show figures
        traj_fig.show()
        states_fig.show()
        inputs_fig.show()
        
        return traj_fig, states_fig, inputs_fig
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

def plot_multiple_simulations(simulation_dirs, labels=None):
    """Plot multiple simulations on the same plot"""
    if labels is None:
        labels = [f"Simulation {i+1}" for i in range(len(simulation_dirs))]
    
    fig = go.Figure()
    
    for sim_dir, label in zip(simulation_dirs, labels):
        states, _, _, _ = load_simulation_data(sim_dir)
        fig.add_trace(go.Scatter3d(
            x=states['x'],
            y=states['y'],
            z=states['z'],
            mode='lines+markers',
            name=label,
            marker=dict(size=2)
        ))
    
    fig.update_layout(
        title="Multiple Simulation Trajectories",
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Z Position',
            aspectmode='data'
        )
    )
    
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot simulation results')
    parser.add_argument('--path', type=str, help='Path to simulation directory')
    parser.add_argument('--compare', type=str, nargs='+', help='Additional simulation paths to compare')
    parser.add_argument('--labels', type=str, nargs='+', help='Labels for comparison plots')
    parser.add_argument('--iteration', type=int, help='Specific iteration number to plot')
    
    args = parser.parse_args()
    
    if args.path:
        # Plot single simulation
        print(f"Plotting simulation from {args.path}")
        if args.iteration is not None:
            # Use specific iteration
            iteration_path = os.path.join(args.path, f'iteration_{args.iteration}')
            if os.path.exists(iteration_path):
                plot_simulation(iteration_path)
            else:
                print(f"Error: Iteration {args.iteration} not found in {args.path}")
        else:
            # Use latest iteration
            plot_simulation(args.path)
    
    if args.compare:
        # Plot multiple simulations
        print("Plotting multiple simulations")
        labels = args.labels if args.labels else [f"Simulation {i+1}" for i in range(len(args.compare))]
        fig = plot_multiple_simulations(args.compare, labels=labels)
        fig.show()
    
    if not args.path and not args.compare:
        # Default behavior: plot most recent simulation
        data_dir = "data/simulation"
        sim_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                    if os.path.isdir(os.path.join(data_dir, d))]
        sim_dirs.sort()
        
        if sim_dirs:
            print(f"Plotting simulation from {sim_dirs[-1]}")
            plot_simulation(sim_dirs[-1])