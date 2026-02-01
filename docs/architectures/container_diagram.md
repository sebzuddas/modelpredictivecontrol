```mermaid
C4Container
    title Container Diagram for MPC Controller System

    Person(user, "User", "Interacts with the MPC system via a web interface or command line.")

    System_Boundary(mpc_system, "MPC Controller System") {
        Container(webapp, "Web Application", "Python/Flask", "Provides a web interface for configuration, running simulations, and visualizing results.")
        Container(mpc_controller_component, "MPC Controller", "Python/CasADi", "Calculates optimal control inputs using Model Predictive Control principles. Includes LQR fallback.")
        Container(system_interface_component, "System Interface", "Python", "Abstracts interaction with the controlled system (either simulation or external API).")
        Container(data_storage, "Data Storage", "Filesystem/JSON/CSV", "Stores simulation results, configuration, and optimization logs.")
        Container(optimization_engine, "Optimization Engine", "Python/DEAP/Multiprocessing", "Performs genetic algorithm-based optimization of MPC parameters (Q and R matrices).")
    }

    System(external_system, "External Controlled System", "Hardware/Software System", "The physical or simulated system (e.g., quadrotor) that the MPC controls.")

    Rel(user, webapp, "Uses", "HTTP")
    Rel(webapp, mpc_controller_component, "Configures and Triggers Runs")
    Rel(webapp, data_storage, "Retrieves Simulation Results from")
    Rel(mpc_controller_component, system_interface_component, "Interacts with", "Python Function Calls")
    Rel(system_interface_component, external_system, "Sends control inputs to and receives state from", "HTTP API or In-Memory Simulation")
    Rel(optimization_engine, mpc_controller_component, "Runs simulations on and tunes parameters for")
    Rel(optimization_engine, data_storage, "Stores Optimization Results and Logs to")
    Rel(webapp, optimization_engine, "Initiates and monitors (indirectly)")

    UpdateRel(mpc_controller_component, data_storage, "Stores Simulation Results")

```