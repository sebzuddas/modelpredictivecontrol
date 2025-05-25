from config import MPCConfig, SystemConfig
from mpc_controller import MPCController, plot_results
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from deap import base, creator, tools, algorithms
import random
import multiprocessing

def evaluate_individual(individual, mpc_config, system_config, timestamp, optimization_dir, iteration):
    """Global evaluation function for parallel processing"""
    Q = np.diag(individual[:mpc_config.Q.shape[0]])
    R = np.diag(individual[mpc_config.Q.shape[0]:mpc_config.Q.shape[0] + mpc_config.R.shape[0]])
    
    mpc_config.Q = Q
    mpc_config.R = R
    
    controller = MPCController(mpc_config, system_config)
    t, x, u, mode = controller.run()
    
    error = np.sum(np.linalg.norm(x - mpc_config.reference_state.reshape(-1, 1), axis=0))
    
    save_simulation_data(t, x, u, mode, mpc_config, 
                        timestamp=timestamp,
                        iteration=iteration,
                        optimization_dir=optimization_dir)
    
    return (error,)

class MPCGeneticOptimizer:
    def __init__(self, pop_size=50, n_gen=20, cx_prob=0.7, mut_prob=0.2):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.iteration = 0
        
        # Define bounds for Q and R matrices
        self.q_bounds = (0.01, 10.0)  # (min, max) for Q matrix elements
        self.r_bounds = (0.01, 10.0)  # (min, max) for R matrix elements
        
        # Create timestamp and directory for this optimization run
        self.optimization_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.optimization_dir = os.path.join('data', 'simulation', self.optimization_timestamp)
        os.makedirs(self.optimization_dir, exist_ok=True)
        
        # Define fitness and individual
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Create toolbox
        self.toolbox = base.Toolbox()
        self._setup_genetic_operators()
    
    def _setup_genetic_operators(self):
        # Attribute generator with bounds
        def attr_q():
            return random.uniform(self.q_bounds[0], self.q_bounds[1])
        
        def attr_r():
            return random.uniform(self.r_bounds[0], self.r_bounds[1])
        
        # Structure initializers
        self.toolbox.register("attr_q", attr_q)
        self.toolbox.register("attr_r", attr_r)
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                            (self.toolbox.attr_q, self.toolbox.attr_r),
                            n=13)  # 9 for Q (states), 4 for R (inputs)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    @staticmethod
    def evaluate_individual(args):
        """Static method for evaluation that can be pickled"""
        individual, mpc_config, system_config, timestamp, optimization_dir, iteration = args
        
        # Extract Q and R matrices from individual
        Q = np.diag(individual[:mpc_config.Q.shape[0]])  # Q diagonal elements
        R = np.diag(individual[mpc_config.Q.shape[0]:mpc_config.Q.shape[0] + mpc_config.R.shape[0]])  # R diagonal elements
        # Verify dimensions
        if R.shape[0] != mpc_config.B.shape[1]:
            raise ValueError(f"R matrix dimensions {R.shape} don't match B matrix input dimensions {mpc_config.B.shape[1]}")
        
        mpc_config.Q = Q
        mpc_config.R = R
        
        try:
            controller = MPCController(mpc_config, system_config)
            t, x, u, mode = controller.run()
            
            error = np.sum(np.linalg.norm(x - mpc_config.reference_state.reshape(-1, 1), axis=0))
            
            save_simulation_data(t, x, u, mode, mpc_config, 
                               timestamp=timestamp,
                               iteration=iteration,
                               optimization_dir=optimization_dir)
            
            return (error,)
            
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"Invalid solution for individual: {e}")
            return (float('inf'),)  # Return worst possible fitness
    
    def optimize(self, mpc_config, system_config):
        """Run genetic algorithm optimization with parallel processing"""
        # Create pool
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        
        # Create initial population
        pop = self.toolbox.population(n=self.pop_size)
        
        # Statistics with safe fitness access
        def safe_fitness(ind):
            if ind.fitness.valid:
                return ind.fitness.values[0]
            return float('inf')  # Return infinity for invalid fitness
        
        stats = tools.Statistics(safe_fitness)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        try:
            # Run genetic algorithm
            for gen in range(self.n_gen):
                # Evaluate individuals
                invalid_ind = [ind for ind in pop if not ind.fitness.valid]
                if invalid_ind:  # Only evaluate if there are invalid individuals
                    fitnesses = pool.map(
                        self.evaluate_individual,
                        [(ind, mpc_config, system_config, self.optimization_timestamp, 
                          self.optimization_dir, self.iteration + i) 
                         for i, ind in enumerate(invalid_ind)]
                    )
                    
                    # Update fitness values correctly
                    for ind, fit in zip(invalid_ind, fitnesses):
                        ind.fitness.values = fit
                    
                    self.iteration += len(invalid_ind)
                
                # Select next generation
                offspring = algorithms.varOr(pop, self.toolbox, 
                                          lambda_=self.pop_size, 
                                          cxpb=self.cx_prob, 
                                          mutpb=self.mut_prob)
                
                # Update population
                pop[:] = offspring
                
                # Record statistics
                record = stats.compile(pop)
                print(f"Generation {gen}: {record}")
            
            # Create logbook
            logbook = tools.Logbook()
            logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            
        finally:
            pool.close()
            pool.join()
        
        # Get best individual
        best_ind = tools.selBest(pop, k=1)[0]
        
        # Save optimization results
        self._save_optimization_results(best_ind, logbook)
        
        return best_ind, logbook
    
    def _save_optimization_results(self, best_ind, logbook):
        """Save optimization results"""
        # Extract Q and R values regardless of fitness
        best_Q = np.diag(best_ind[:9]).tolist()
        best_R = np.diag(best_ind[9:]).tolist()
        
        # Handle fitness value safely
        try:
            best_fitness = best_ind.fitness.values[0]
        except (IndexError, AttributeError):
            print("Warning: Best individual has invalid fitness")
            best_fitness = float('inf')
        
        results = {
            'timestamp': self.optimization_timestamp,
            'best_fitness': best_fitness,
            'best_Q': best_Q,
            'best_R': best_R,
            'optimization_history': logbook
        }
        
        # Save to JSON file
        results_file = os.path.join(self.optimization_dir, 'optimization_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)

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

def save_simulation_data(t, x, u, mode, config, timestamp=None, iteration=None, optimization_dir=None):
    """Save simulation data to a structured format"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create optimization directory if it doesn't exist
    if optimization_dir is None:
        optimization_dir = os.path.join('data', 'simulation', timestamp)
    os.makedirs(optimization_dir, exist_ok=True)
    
    # Create iteration directory
    if iteration is not None:
        data_dir = os.path.join(optimization_dir, f'iteration_{iteration}')
    else:
        data_dir = os.path.join(optimization_dir, 'single_run')
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

def run_optimization(pop_size=2, n_gen=2):
    """Run the genetic algorithm optimization"""
    print("Starting genetic algorithm optimization...")
    print(f"Using {multiprocessing.cpu_count()-4} CPU cores")
    
    # Create configurations
    mpc_config = MPCConfig()
    system_config = SystemConfig()
    
    # Create optimizer
    optimizer = MPCGeneticOptimizer(pop_size=pop_size, n_gen=n_gen)
    
    # Run optimization
    best_ind, logbook = optimizer.optimize(mpc_config, system_config)
    
    # Safely access fitness value
    try:
        best_fitness = best_ind.fitness.values[0]
    except (IndexError, AttributeError):
        print("Warning: Best individual has invalid fitness")
        best_fitness = float('inf')
    
    print(f"Best fitness: {best_fitness}")
    print(f"Best Q: {np.diag(best_ind[:mpc_config.Q.shape[0]])}")
    print(f"Best R: {np.diag(best_ind[mpc_config.Q.shape[0]:mpc_config.Q.shape[0] + mpc_config.R.shape[0]])}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MPC controller in different modes')
    parser.add_argument('mode', choices=['simulation', 'api', 'optimization'], 
                      help='Mode to run: simulation, api, or optimization')
    parser.add_argument('--pop-size', type=int, default=20,
                      help='Population size for optimization (default: 20)')
    parser.add_argument('--n-gen', type=int, default=50, 
                      help='Number of generations for optimization (default: 50)')
    
    args = parser.parse_args()
    
    if args.mode == 'simulation':
        run_simulation()
    elif args.mode == 'api':
        run_api()
    elif args.mode == 'optimization':
        multiprocessing.set_start_method('spawn')
        run_optimization(pop_size=args.pop_size, n_gen=args.n_gen)