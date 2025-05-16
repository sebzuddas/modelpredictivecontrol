import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import os
import argparse
from datetime import datetime

def load_simulation_data(simulation_dir):
    """Load simulation data from a directory"""
    # Check if this is an iteration directory
    if os.path.basename(simulation_dir).startswith('iteration_'):
        # Direct path to iteration data
        states = pd.read_csv(os.path.join(simulation_dir, 'states.csv'))
        inputs = pd.read_csv(os.path.join(simulation_dir, 'inputs.csv'))
        time = pd.read_csv(os.path.join(simulation_dir, 'time.csv'))
        config = json.load(open(os.path.join(simulation_dir, 'config.json')))
    else:
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
        print(f"Loading data from: {sim_dir}")
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

def create_trajectory_video(simulation_dir, output_file="trajectory_animation.html"):
    """Create an animated visualization of all iterations"""
    # Find all iteration directories
    iteration_dirs = [d for d in os.listdir(simulation_dir) 
                     if os.path.isdir(os.path.join(simulation_dir, d)) 
                     and d.startswith('iteration_')]
    iteration_dirs.sort(key=lambda x: int(x.split('_')[1]))  # Sort by iteration number
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each iteration
    for i, iter_dir in enumerate(iteration_dirs):
        iter_path = os.path.join(simulation_dir, iter_dir)
        states, _, _, _ = load_simulation_data(iter_path)
        
        # Add trace for this iteration
        fig.add_trace(go.Scatter3d(
            x=states['x'],
            y=states['y'],
            z=states['z'],
            mode='lines+markers',
            name=f'Iteration {iter_dir.split("_")[1]}',
            visible=False,  # Start with all traces hidden
            marker=dict(size=2),
            line=dict(width=2)
        ))
    
    # Make first trace visible
    fig.data[0].visible = True
    
    # Create animation buttons
    buttons = []
    for i in range(len(fig.data)):
        button = dict(
            method='update',
            args=[{'visible': [False] * len(fig.data)},
                  {'title': f'Iteration {iteration_dirs[i].split("_")[1]}'}],
            label=f'Iteration {iteration_dirs[i].split("_")[1]}'
        )
        button['args'][0]['visible'][i] = True  # Make i-th trace visible
        buttons.append(button)
    
    # Add play button
    buttons.append(
        dict(
            method='animate',
            args=[None, dict(
                frame=dict(duration=1000, redraw=True),
                fromcurrent=True,
                mode='immediate',
                transition=dict(duration=500)
            )],
            label='Play'
        )
    )
    
    # Create frames for animation
    frames = []
    for i in range(len(fig.data)):
        frame = go.Frame(
            data=[go.Scatter3d(
                x=fig.data[i].x,
                y=fig.data[i].y,
                z=fig.data[i].z,
                mode='lines+markers',
                name=f'Iteration {iteration_dirs[i].split("_")[1]}',
                marker=dict(size=2),
                line=dict(width=2)
            )],
            name=f'Iteration {iteration_dirs[i].split("_")[1]}'
        )
        frames.append(frame)
    
    # Update layout
    fig.update_layout(
        title='Drone Trajectory Evolution',
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Z Position',
            aspectmode='data'
        ),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=buttons,
            direction='right',
            pad=dict(t=10),
            x=0.1,
            y=0,
            xanchor='right',
            yanchor='top'
        )],
        sliders=[dict(
            currentvalue=dict(prefix='Iteration: '),
            pad=dict(t=50),
            steps=[dict(
                method='animate',
                args=[[f'frame{i}'], dict(
                    mode='immediate',
                    frame=dict(duration=1000, redraw=True),
                    transition=dict(duration=500)
                )],
                label=f'Iteration {iteration_dirs[i].split("_")[1]}'
            ) for i in range(len(fig.data))]
        )]
    )
    
    # Add frames to figure
    fig.frames = frames
    
    # Save as interactive HTML
    fig.write_html(output_file)
    print(f"Animation saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot simulation results')
    parser.add_argument('--path', type=str, help='Path to simulation directory')
    parser.add_argument('--compare', type=int, nargs='+', help='Iteration numbers to compare')
    parser.add_argument('--all', action='store_true', help='Plot all iterations')
    parser.add_argument('--labels', type=str, nargs='+', help='Labels for comparison plots')
    parser.add_argument('--sim-dir', type=str, default='data/simulation', 
                       help='Base simulation directory (default: data/simulation)')
    parser.add_argument('--video', action='store_true', help='Create animated visualization')
    parser.add_argument('--output', type=str, default='trajectory_animation.html',
                       help='Output file for animation (default: trajectory_animation.html)')
    
    args = parser.parse_args()
    
    if args.path:
        # Plot single simulation
        print(f"Plotting simulation from {args.path}")
        plot_simulation(args.path)
    
    if args.compare or args.all:
        # Get the most recent simulation directory
        sim_dirs = [os.path.join(args.sim_dir, d) for d in os.listdir(args.sim_dir) 
                    if os.path.isdir(os.path.join(args.sim_dir, d))]
        sim_dirs.sort()
        
        if not sim_dirs:
            print("No simulation directories found")
            exit(1)
            
        latest_sim_dir = sim_dirs[-1]
        print(f"Using simulation directory: {latest_sim_dir}")
        
        if args.all:
            # Find all iteration directories
            iteration_dirs = [d for d in os.listdir(latest_sim_dir) 
                            if os.path.isdir(os.path.join(latest_sim_dir, d)) 
                            and d.startswith('iteration_')]
            iteration_dirs.sort(key=lambda x: int(x.split('_')[1]))  # Sort by iteration number
            
            # Create paths for all iterations
            iteration_paths = [os.path.join(latest_sim_dir, d) for d in iteration_dirs]
            labels = [f"Iteration {d.split('_')[1]}" for d in iteration_dirs]
            
        else:
            # Create paths for specified iterations
            iteration_paths = [os.path.join(latest_sim_dir, f'iteration_{i}') for i in args.compare]
            labels = args.labels if args.labels else [f"Iteration {i}" for i in args.compare]
        
        # Verify all iterations exist
        for path in iteration_paths:
            if not os.path.exists(path):
                print(f"Error: Iteration directory not found: {path}")
                exit(1)
        
        # Plot multiple iterations
        print(f"Plotting {len(iteration_paths)} iterations")
        fig = plot_multiple_simulations(iteration_paths, labels=labels)
        fig.show()
    
    if args.video:
        # Get the most recent simulation directory
        sim_dirs = [os.path.join(args.sim_dir, d) for d in os.listdir(args.sim_dir) 
                    if os.path.isdir(os.path.join(args.sim_dir, d))]
        sim_dirs.sort()
        
        if not sim_dirs:
            print("No simulation directories found")
            exit(1)
            
        latest_sim_dir = sim_dirs[-1]
        print(f"Creating animation from {latest_sim_dir}")
        create_trajectory_video(latest_sim_dir, args.output)
    
    if not args.path and not args.compare and not args.all:
        # Default behavior: plot most recent simulation
        data_dir = "data/simulation"
        sim_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                    if os.path.isdir(os.path.join(data_dir, d))]
        sim_dirs.sort()
        
        if sim_dirs:
            print(f"Plotting simulation from {sim_dirs[-1]}")
            plot_simulation(sim_dirs[-1])