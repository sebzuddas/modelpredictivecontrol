# TO-DO List
## MPC Controller & Simulation
- Include disturbances in the simulation
  - white noise (sensor noise)
  - non-linear disturbances
- Check the $dt$ parameter and how it relates to the model. 
- Generate the controller etc from a yaml config file. 
- For the rotational components of the state space, generate a plot on polar coordinates.
- Implement disturbance rejection and reference tracking
- Implement an EKF for state estimation 

## Parameter Optimization
- Demonstrate the population generated. 
- Place into yaml files
- Show to user to check and edit if needed
- allow user to select generations and population
- Test for exception handling for pushing to HPC
- Running populations relative to number of cores
- Add early stopping criteria

## Mission Capabilities
- add reference tracking
- track two different references
  - track the different references when different conditions are met. 
  - finite-state machines?