from manim import *
import pandas as pd
import os
import json
import numpy as np

class DroneTrajectory(ThreeDScene):
    def construct(self):
        # Load simulation data
        def load_simulation_data(simulation_dir):
            """Load simulation data from a directory"""
            # Find all iteration directories
            iteration_dirs = [d for d in os.listdir(simulation_dir) 
                            if os.path.isdir(os.path.join(simulation_dir, d)) 
                            and d.startswith('iteration_')]
            iteration_dirs.sort(key=lambda x: int(x.split('_')[1]))
            
            all_iterations = []
            for iter_dir in iteration_dirs:
                iter_path = os.path.join(simulation_dir, iter_dir)
                states = pd.read_csv(os.path.join(iter_path, 'states.csv'))
                all_iterations.append(states)
            
            return all_iterations

        def normalize_data(iterations):
            """Normalize all trajectory data to fit within axes, keeping origin at (0,0,0)"""
            # Find global min and max for each dimension
            x_min = min(df['x'].min() for df in iterations)
            x_max = max(df['x'].max() for df in iterations)
            y_min = min(df['y'].min() for df in iterations)
            y_max = max(df['y'].max() for df in iterations)
            z_min = min(df['z'].min() for df in iterations)
            z_max = max(df['z'].max() for df in iterations)
            
            # Find the maximum range to maintain aspect ratio
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            max_range = max(x_range, y_range, z_range)
            
            # Normalize each iteration
            normalized_iterations = []
            for df in iterations:
                normalized_df = df.copy()
                # Scale to [-0.9, 0.9] range without centering
                normalized_df['x'] = 1.8 * df['x'] / max_range
                normalized_df['y'] = 1.8 * df['y'] / max_range
                normalized_df['z'] = 1.8 * df['z'] / max_range
                normalized_iterations.append(normalized_df)
            
            return normalized_iterations

        # Load data from most recent simulation
        data_dir = "data/simulation"
        sim_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                    if os.path.isdir(os.path.join(data_dir, d))]
        sim_dirs.sort()
        latest_sim_dir = sim_dirs[-1]
        
        print(f"Loading data from {latest_sim_dir}")
        iterations = load_simulation_data(latest_sim_dir)
        
        # Normalize the data
        normalized_iterations = normalize_data(iterations)
        
        # Get simulation directory name for file naming
        sim_name = os.path.basename(latest_sim_dir)
        
        # Set up the 3D scene with initial camera orientation
        self.set_camera_orientation(phi=60 * DEGREES, theta=45 * DEGREES, zoom=0.8)
        
        # Start constant camera rotation
        self.begin_ambient_camera_rotation(rate=0.2)
        
        # Create axes
        axes = ThreeDAxes(
            x_range=[-1, 1, 0.2],
            y_range=[-1, 1, 0.2],
            z_range=[-1, 1, 0.2],
            x_length=6,
            y_length=6,
            z_length=6,
        )
        self.add(axes)
        
        # Add labels
        x_label = axes.get_x_axis_label("x")
        y_label = axes.get_y_axis_label("y")
        self.add(x_label, y_label)
        
        # Create trajectories for each iteration
        for i, states in enumerate(normalized_iterations):
            # Update z-label with iteration number
            z_label = axes.get_z_axis_label(f"z (Iteration {i})")
            self.add(z_label)
            
            # Create points for the trajectory
            points = []
            for _, row in states.iterrows():
                point = [row['x'], row['y'], row['z']]
                points.append(point)
            
            # Create the trajectory line
            trajectory = VMobject()
            trajectory.set_points_as_corners(points)
            trajectory.set_stroke(width=2, color=BLUE)
            
            # Create the drone dot (larger and more visible)
            drone = Sphere(radius=0.05, color=RED)
            drone.move_to(points[0])
            
            # Create iteration text
            iteration_text = Text(f"Iteration: {i}", font_size=24)
            iteration_text.to_corner(UL)
            
            # Animate the trajectory
            self.play(
                Create(trajectory),
                FadeIn(drone),
                Write(iteration_text),
                run_time=2
            )
            
            # Animate the drone moving along the trajectory
            self.play(
                MoveAlongPath(drone, trajectory),
                run_time=10
            )
            
            # Fade out for next iteration
            self.play(
                FadeOut(trajectory),
                FadeOut(drone),
                FadeOut(iteration_text),
                FadeOut(z_label),
                run_time=1
            )
        
        # Keep rotating for a moment after the last iteration
        self.wait(2)
        
        # Stop camera rotation at the end
        self.stop_ambient_camera_rotation()

if __name__ == "__main__":
    # Get the most recent simulation directory
    data_dir = "data/simulation"
    sim_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                if os.path.isdir(os.path.join(data_dir, d))]
    sim_dirs.sort()
    latest_sim_dir = sim_dirs[-1]
    sim_name = os.path.basename(latest_sim_dir)
    
    # Render the scene with custom output name
    os.system(f"manim -pql animation.py DroneTrajectory -o {sim_name}_trajectory")