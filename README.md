# TO-DO List
## MPC Controller & Simulation
- Include disturbances in the simulation
  - white noise (sensor noise)
  - non-linear disturbances
- Check the $dt$ parameter and how it relates to the model. 
- Generate the controller etc from a yaml config file. 

## Parameter Optimization
- Generate LHS
- Place into yaml files
- Show to user to check and edit if needed
- allow user to select generations and population
- Test for exception handling for pushing to HPC

## Mission Capabilities
- add reference tracking
- track two different references
  - track the different references when different conditions are met. 
  - finite-state machines?
  - 