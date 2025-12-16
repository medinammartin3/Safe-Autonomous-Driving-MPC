# Safe-Autonomous-Driving-MPC

**A Hierarchical Control Architecture for Safe Urban Autonomous Driving**

## Project Overview

**Safe-Autonomous-Driving-MPC** is a Python-based simulation framework that implements a two-stage Model Predictive Control (MPC) architecture. The system is designed to navigate an autonomous vehicle through realistic urban environments by first generating an optimal reference trajectory offline, and then tracking it in real-time while strictly respecting speed limits, handling static and dynamic obstacles, and adhering to vehicle dynamic constraints.

By separating **path planning** (Global Routing) and **trajectory planning** (Non-Linear MPC) from **real-time tracking** (Linearized MPC), the system balances computational efficiency with safety, enabling responsive behavior in dynamic environments.

## Key Features

* **Real-World Map Integration**: Utilizes the **GraphHopper API** to fetch real-world road networks, speed limits, and waypoints, processing them into a continuous cubic spline.
* **Two-Stage MPC Architecture**:
    * **Offline Trajectory Planner**: A Non-Linear MPC (using Hermite-Simpson collocation) that generates a time-optimal velocity profile respecting road curvature and passenger comfort.
    * **Real-Time Tracker**: A Linearized MPC (using Explicit-Euler collocation) that tracks the reference trajectory and adjusts control inputs (steering/acceleration) in real-time.
* **Dynamic Obstacle Avoidance**: Features a Finite State Machine (FSM) to handle interaction with traffic lights (static) and moving vehicles (dynamic).
* **Safety Constraints**: Enforces hard constraints for collision avoidance, safe lane margins, and speed limits.
* **Rich Visualization**: Includes tools for animating the driving simulation and plotting detailed telemetry data (velocity, lateral deviation, control inputs).

## Project Structure

```text
Safe-Autonomous-Driving-MPC/
├── animations/
│   ├── driving_simulation.gif      # Corresponds to trajectory3.json (Long complex scenario)
│   └── driving_simulation2.gif     # Corresponds to trajectory2.json (Medium complex scenario)
├── trajectories/
│   ├── trajectory1.json            # Simple 300m scenario suitable for testing without obstacles
│   ├── trajectory2.json            # Complex 1300m path suitable for testing with obstacles (matches driving_simulation2.gif)
│   └── trajectory3.json            # Long 3000m+ path suitable for complex validation with obstacles (matches driving_simulation.gif)
├── .env                            # (Create this) Stores your GraphHopper API Key
├── path_planning.py                # Interfaces with GraphHopper API to retrieve route data and build splines
├── trajectory_planning.py          # Offline Non-Linear MPC for generating optimal reference path
├── trajectory_tracking.py          # Real-time Linear MPC for tracking and obstacle avoidance
├── trajectory_loader.py            # Utility to load and interpolate the offline JSON trajectory
├── visualizations.py               # Matplotlib scripts for plotting results and animations
├── sanity_checks.py                # Verification modules for safety and feasibility limits
├── poster.pdf                      # Project presentation poster
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

### Key Modules
* **`path_planning.py`**: Processes the raw map data into a continuous cubic spline for the solver.
* **`trajectory_planning.py`**: Solves the NLP (Non-Linear Programming) problem to create a reference trajectory. It minimizes travel time and jerk while respecting global speed limits.
* **`trajectory_tracking.py`**: The main simulation loop. It loads the reference trajectory and runs the tracking MPC, handling the `ObstaclesFSM` logic for traffic lights and dynamic vehicles.
* **`sanity_checks.py`**: Ensures the vehicle respects physics, stays within lane boundaries, and maintains safe distances during simulation.

## Getting Started

Follow these steps to set up the environment and run the simulation.

### 1. Prerequisites
* Python 3.8+
* A **GraphHopper** API Key (Free tier available).

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/medinammartin3/Safe-Autonomous-Driving-MPC
cd Safe-Autonomous-Driving-MPC
pip install -r requirements.txt
```

### 3. Environment Configuration

To fetch real-world road data, you must provide a GraphHopper API key.

1.  Create a file named `.env` in the root directory.
2.  Add your API key using the variable name `GRAPHHOPPER_API_KEY`:

```env
GRAPHHOPPER_API_KEY=your_actual_api_key_here
```

## Usage

The architecture works in two distinct phases: **Planning** (Offline) and **Tracking** (Online).

### Phase 1: Trajectory Planning (Offline)
This step fetches the route from GraphHopper, optimizes the velocity profile using Non-Linear MPC, and saves the result to a JSON file.

1.  Open `trajectory_planning.py`.
2.  Define your `start` and `end` locations (some default examples corresponding to the existing JSON files are provided).
3.  **Important** - If generating a new custom trajectory: 
      * Uncomment the JSON saving block to save your reference path.
      * Ensure these location strings match valid GraphHopper addresses or point of interest names.
4.  Run the optimizer:

```bash
python trajectory_planning.py
```

*Output: This will generate a visualization of the optimal path and save the data to `trajectories/trajectory.json`.*

### Phase 2: Trajectory Tracking (Real-Time Simulation)
This step loads the generated trajectory and simulates the vehicle driving along it, reacting to obstacles in real-time.

1.  Open `trajectory_tracking.py`.
2.  Ensure `trajectory_file` points to the desired JSON (e.g., `trajectories/trajectory2.json`).
3.  Configure the `ObstaclesFSM` in the `__main__` block if you want to enable/disable specific obstacles.
4.  Run the tracker:

```bash
python trajectory_tracking.py
```

*Output: The script will print simulation steps to the console. Upon completion, it will launch a Matplotlib window showing performance graphs and a 2D driving animation.*

## Visualizing Results

The simulation automatically generates plots for:
* Tracking errors (Lateral/Orientation)
* Control inputs (Curvature rate/Acceleration)
* Distance to obstacles and traffic light status

To save animations, ensure the `visualizations.py` script is configured to save to the `animations/` folder.

## Authors

* **[Martin Medina](https://github.com/medinammartin3)**
* **[Étienne Mitchell-Bouchard](https://github.com/darkzant)**

*University of Montreal / Mila*