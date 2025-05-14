// Chart instances
let stateChart = null;
let inputChart = null;
let modeChart = null;

// Initialize charts
function initCharts() {
    // State trajectories chart
    const stateCtx = document.getElementById('statePlot').getContext('2d');
    stateChart = new Chart(stateCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Position',
                    borderColor: '#007bff',
                    data: [],
                    fill: false
                },
                {
                    label: 'Velocity',
                    borderColor: '#28a745',
                    data: [],
                    fill: false
                },
                {
                    label: 'Reference',
                    borderColor: '#dc3545',
                    data: [],
                    fill: false,
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time (s)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'State'
                    }
                }
            }
        }
    });

    // Control input chart
    const inputCtx = document.getElementById('inputPlot').getContext('2d');
    inputChart = new Chart(inputCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Control Input',
                borderColor: '#6f42c1',
                data: [],
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time (s)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Input'
                    }
                }
            }
        }
    });

    // Control mode chart
    const modeCtx = document.getElementById('modePlot').getContext('2d');
    modeChart = new Chart(modeCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Control Mode',
                borderColor: '#fd7e14',
                data: [],
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time (s)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Mode'
                    },
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}

// Load current configuration
async function loadConfig() {
    try {
        const response = await fetch('/api/config');
        const config = await response.json();
        
        // Update form fields
        document.getElementById('systemType').value = config.system.system_type;
        document.getElementById('dt').value = config.mpc.dt;
        document.getElementById('horizon').value = config.mpc.N;
        document.getElementById('simTime').value = config.mpc.simulation_time;
        document.getElementById('Q1').value = config.mpc.Q[0][0];
        document.getElementById('Q2').value = config.mpc.Q[1][1];
        document.getElementById('R').value = config.mpc.R[0][0];
        document.getElementById('vMin').value = config.mpc.x_min[1];
        document.getElementById('vMax').value = config.mpc.x_max[1];
        document.getElementById('uMin').value = config.mpc.u_min[0];
        document.getElementById('uMax').value = config.mpc.u_max[0];
        document.getElementById('initPos').value = config.mpc.initial_state[0];
        document.getElementById('refPos').value = config.mpc.reference_state[0];
    } catch (error) {
        console.error('Error loading configuration:', error);
        alert('Failed to load configuration');
    }
}

// Save configuration
async function saveConfig() {
    const config = {
        mpc: {
            dt: parseFloat(document.getElementById('dt').value),
            N: parseInt(document.getElementById('horizon').value),
            Q: [[parseFloat(document.getElementById('Q1').value), 0], [0, parseFloat(document.getElementById('Q2').value)]],
            R: [[parseFloat(document.getElementById('R').value)]],
            x_min: [null, parseFloat(document.getElementById('vMin').value)],
            x_max: [null, parseFloat(document.getElementById('vMax').value)],
            u_min: [parseFloat(document.getElementById('uMin').value)],
            u_max: [parseFloat(document.getElementById('uMax').value)],
            simulation_time: parseFloat(document.getElementById('simTime').value),
            initial_state: [parseFloat(document.getElementById('initPos').value), 0],
            reference_state: [parseFloat(document.getElementById('refPos').value), 0]
        },
        system: {
            system_type: document.getElementById('systemType').value
        }
    };

    try {
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });
        
        if (response.ok) {
            alert('Configuration saved successfully');
        } else {
            throw new Error('Failed to save configuration');
        }
    } catch (error) {
        console.error('Error saving configuration:', error);
        alert('Failed to save configuration');
    }
}

// Run simulation
async function runSimulation() {
    try {
        const response = await fetch('/api/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        console.log('Received data:', data);  // Debug log
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to run simulation');
        }
        
        updatePlots(data);
    } catch (error) {
        console.error('Error running simulation:', error);
        alert(error.message);
    }
}

// Update plots with simulation results
function updatePlots(results) {
    const t = results.t;
    const x = results.x;
    const u = results.u;
    const mode = results.mode;
    const reference_state = results.reference_state;
    // Update state trajectories
    stateChart.data.labels = t;
    stateChart.data.datasets[0].data = x[0]; // Position trajectory
    stateChart.data.datasets[1].data = x[1]; // Velocity trajectory
    stateChart.data.datasets[2].data = t.map(() => reference_state[0]); // Reference
    stateChart.update();
    
    // Update control input
    inputChart.data.labels = t;
    inputChart.data.datasets[0].data = u[0];
    inputChart.update();
    
    // Update control mode
    modeChart.data.labels = t;
    modeChart.data.datasets[0].data = mode;
    modeChart.update();
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    loadConfig();
    
    document.getElementById('saveConfig').addEventListener('click', saveConfig);
    document.getElementById('runSimulation').addEventListener('click', runSimulation);
}); 