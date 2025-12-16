import numpy as np
from sanity_checks import trajectory_tracking_check
from trajectory_loader import TrajectoryLoader
from scipy.optimize import minimize
import time


class TrajectoryTracker:
    """
    Real-time MPC for tracking the optimal reference trajectory.
    """
    def __init__(self, X_ref=None):
        # Reference trajectory
        self.X_ref = X_ref

        # MPC parameters
        self.dt = 0.2  # Time step (seconds)
        self.N = 5     # Horizon (1 second)

        # Vehicle constraints
        """
        u1_min = Max Steering Rate (Right)
        u1_max = Min Steering Rate (Left)

        u2_min = Max Braking Deceleration
        u2_max = Min Engine Acceleration

        k_min = Max Curvature (Right Turn)
        k_max = Max Curvature (Left Turn)
        """
        self.u_min = np.array([-0.6, -5.0])
        self.u_max = np.array([0.6, 4.0])
        self.vehicle_radius = 1.0

        # Weights
        self.w_d = 10.0
        self.w_o = 10.0
        self.w_v = 5.0
        self.w_u1 = 0.5
        self.w_u2 = 0.5

        # Safety constraints
        self.obstacle_safety_distance = 5.0  # Minimum safety distance with obstacles (vehicles/traffic lights)
        self.max_time_2_obs = 1.5  # Minimum time gap between the vehicle and the obstacle
        self.wheelbase = 2.8  # Distance between front and back wheels
        self.lane_width = 3.0
        self.safe_lane_margin = 0.1  # Safety distance with lane limits


    def dynamics(self, x, u, k_ref):
        """
        Linearized system dynamics to reach feasible solutions faster (real-time).
        """
        s, d, o, k, v = x
        u1, u2 = u

        # Dynamics
        s_dot = v
        d_dot = v * o
        o_dot = v * (k - k_ref)
        k_dot = u1
        v_dot = u2

        # Derivative
        x_dot = np.array([s_dot, d_dot, o_dot, k_dot, v_dot])

        return x_dot

    def unpack(self, U_flat):
        """
        Convert the flat controls vector U_flat into matrix of shape (N, 2).
        """
        controls_num = 2
        N = self.N

        U = U_flat.reshape(N, controls_num)  # Matrix of shape (N, 2)
        return U

    @staticmethod
    def pack(U):
        """
        Convert the controls matrix U into one flat vector U_flat.
        """
        U_flat = U.ravel()  # N*2 vector
        return U_flat

    def predict(self, x0, U_flat):
        """
        Simulates the vehicle dynamics forward in time for N steps using Explicit-Euler collocation.
        """
        U = self.unpack(U_flat)

        # Prediction array initialization
        X_pred = np.zeros((self.N + 1, 5))

        # Current state of the vehicle
        X_pred[0] = x0
        curr_x = x0.copy()

        for k in range(self.N):
            # Current position (s)
            curr_s = curr_x[0]
            # Reference state at s
            X_ref_s = self.X_ref.get_state(curr_s)
            # Curvature of the road
            k_ref = X_ref_s[3]
            # Dynamics
            x_dot = self.dynamics(curr_x, U[k], k_ref)
            # Explicit-Euler collocation
            curr_x = curr_x + self.dt * x_dot
            # Store the predicted state for the next step
            X_pred[k + 1] = curr_x

        return X_pred

    def cost(self, U_flat, x0):
        """
        Modified MPC objective function for the linearized dynamics.
        """
        U = self.unpack(U_flat)

        # Generate the predicted trajectory
        X_pred = self.predict(x0, U_flat)

        cost = 0.0

        # State cost : Penalize deviations from the reference trajectory
        for k in range(1, self.N + 1):
            s, d, o, k, v = X_pred[k]

            # Reference state at position s (X = [s, d, o, k, v])
            X_ref_s = self.X_ref.get_state(s)
            d_ref = X_ref_s[1]  # Lateral deviation
            o_ref = X_ref_s[2]  # Orientation deviation
            v_ref = X_ref_s[4]  # Speed

            # Cost terms
            cost += self.w_d * (d - d_ref) ** 2  # Lateral deviation error
            cost += self.w_o * (o - o_ref) ** 2  # Orientation deviation error
            cost += self.w_v * (v - v_ref) ** 2  # Speed error

        # Controls cost : Penalize aggressive inputs (Comfort)
        for k in range(self.N):
            # Agressive changes in steering curvature
            u1 = U[k, 0]
            cost += self.w_u1 * u1 ** 2

            # Agressive acceleration/braking
            u2 = U[k, 1]
            cost += self.w_u2 * u2 ** 2

        return cost


    def constraints(self, x0, obstacles):
        """
        Safety hard constraints.
        Handles both dynamic (cars) and static (traffic lights) obstacles.

        Parameters:
            - obstacles : list of dicts {'s': obstacle position, 'v': obstacle speed}
        """

        def constraints_wrapper(U_flat):
            X_pred = self.predict(x0, U_flat)
            violations = []

            # Safe lane boundaries
            safe_lane_distance = self.lane_width / 2.0 - self.vehicle_radius - self.safe_lane_margin

            for k in range(1, self.N + 1):
                s = X_pred[k, 0]
                d = X_pred[k, 1]
                o = X_pred[k, 2]
                v = X_pred[k, 4]

                # --- Lane constraints ---

                # Lateral deviation bound (Center)
                violations.append(safe_lane_distance - d)
                violations.append(d + safe_lane_distance)

                # Front corner bound approx
                val_front = d + (self.wheelbase / 2.0) * o
                violations.append(safe_lane_distance - val_front)
                violations.append(val_front + safe_lane_distance)

                # Full length bound approx
                val_full = d + self.wheelbase * o
                violations.append(safe_lane_distance - val_full)
                violations.append(val_full + safe_lane_distance)

                # --- Obstacle constraints ---
                for obs in obstacles:
                    # Predict obstacle location
                    s_obs_pred = obs['s'] + obs['v'] * (k * self.dt)

                    # Distance gap between the vehicle and the obstacle
                    s_2_obs = s_obs_pred - s

                    # Safe following distance
                    s_safe = max(self.obstacle_safety_distance, v * self.max_time_2_obs)

                    violations.append(s_2_obs - s_safe)

                # Velocity constraint (non-negative)
                violations.append(v)

            return np.array(violations)

        return {'type': 'ineq', 'fun': constraints_wrapper}

    def solve(self, x0, obstacles):
        """
        Solve the linearized optimization problem to find the best controls.
        We return only the first optimal control to apply the receding horizon principle.

        Parameters:
            - x0 : current state of the vehicle.
            - obstacles : list of dicts {'s': obstacle position, 'v': obstacle speed}
        """

        # Warm start states with current distance and speed
        u_init = []
        s_curr = x0[0]
        v_curr = x0[4]

        # Warm start controls (braking)
        should_brake = False
        for _ in range(self.N):
            for obs in obstacles:
                if (obs['s'] - s_curr) < 40.0:  # Obstacle is close (<40m)
                    should_brake = True

            # Reference controls at distance s
            U_ref_s = self.X_ref.get_control(s_curr)

            if should_brake:
                u1_ref_s = U_ref_s[0]
                u_init.append([u1_ref_s, -2.0])  # Warm start with braking control
            else:
                u_init.append(U_ref_s)  # Warm sart with reference controls

            # Approximate distance for next step guess
            s_curr += v_curr * self.dt
        U_init = self.pack(np.array(u_init))

        # Controls limits and constraints
        bounds = [(self.u_min[0], self.u_max[0]), (self.u_min[1], self.u_max[1])] * self.N
        constraints = self.constraints(x0, obstacles)

        # Solve with SLSQP
        start_t = time.time()
        solution = minimize(self.cost, U_init, args=(x0,), method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'ftol': 1e-3, 'disp': False, 'maxiter': 15})
        end_t = time.time()
        solve_time = end_t - start_t

        U_sol = self.unpack(solution.x)
        pred_X = self.predict(x0, solution.x)

        return U_sol[0], pred_X, solve_time  # Return only first optimal control


class ObstaclesFSM:
    """
    Finite State Machine (FSM) for handling different obstacle scenarios and lifecycles during simulation.
    It monitors the vehicle's position to trigger and manage states for obstacles (traffic lights and dynamic vehicles).

    Each obstacle has its own FSM (that runs in parallel) defined as follows:

    * Dynamic Obstacle (External Vehicle) FSM:
        - States:
            1. WAITING: Waiting for the vehicle to reach the trigger position.
            2. ACTIVE: The obstacle is spawned and moves at constant velocity along the path.
            3. COMPLETED: The obstacle has completed its lifecycle and is removed.

    * Traffic Light (Static Obstacle) FSM:
        - States:
            1. RED_APPROACH: The light is RED. A static obstacle (virtual wall) is enforced at the traffic light position.
            2. RED_WAITING: The vehicle has fully stopped at the traffic light. A timer counts the wait duration.
            3. GREEN: The timer reached the waiting time. The static obstacle is removed, allowing the vehicle to proceed.
    """
    def __init__(self, dynamic_obstacle=False, traffic_light=False):
        """
        Initializes the parameters for the world interaction scenarios.
        """
        self.dynamic_obstacle = dynamic_obstacle
        self.traffic_light = traffic_light

        """ --- Good Configuration for trajectory2.json --- """
        # Dynamic Obstacle parameters
        self.obs_trigger_s = 710.0      # Position of the vehicle when the obstacle appear
        self.obs_start_s = 780.0        # Start position of the obstacle
        self.obs_v = 4.0                # Speed [m/s] of the obstacle (constant)
        self.obs_end_s = 1050.0         # End position of the obstacle (disappears)
        self.obs_active = False         # Is the obstacle currently on road ?
        self.obs_s = self.obs_start_s   # Current position of the obstacle
        self.obs_has_triggered = False  # Has already the obstacle been triggered ?

        # Traffic Light parameters
        self.tl_pos = 550.0           # Position of the traffic light
        self.tl_trigger_s = 100.0     # Interaction starts 100m before light
        self.tl_stop_duration = 20.0  # Duration of the red light once the vehicle has fully stopped
        self.tl_state = 'RED'         # Traffic Light color [RED, GREEN]
        self.tl_timer = 0.0           # Timer for switching to GREEN
        self.tl_waiting = False       # Is the vehicle waiting at the light ?


        """ --- Good Configuration for trajectory3.json --- """
        # # Dynamic Obstacle parameters
        # self.obs_trigger_s = 5.0        # Position of the vehicle when the obstacle appear
        # self.obs_start_s = 150.0        # Start position of the obstacle
        # self.obs_v = 4.0                # Speed of the obstacle (constant)
        # self.obs_end_s = 850.0          # End position of the obstacle (disappears)
        # self.obs_active = False         # Is the obstacle currently on road ?
        # self.obs_s = self.obs_start_s   # Current position of the obstacle
        # self.obs_has_triggered = False  # Has already the obstacle been triggered ?
        #
        # # Traffic Light parameters
        # self.tl_pos = 2000.0          # Position of the traffic light
        # self.tl_trigger_s = 100.0     # Interaction starts 100m before light
        # self.tl_stop_duration = 20.0  # Duration of the red light once the vehicle has fully sopped
        # self.tl_state = 'RED'         # Traffic Light color [RED, GREEN]
        # self.tl_timer = 0.0           # Timer for switching to GREEN
        # self.tl_waiting = False       # Is the vehicle waiting at the light ?


    def update(self, dt, s, v):
        """
        Updates the FSM state based on time (dt) and vehicle status.
        Return:
            - active_obstacles : list of obstacles that the tracking MPC should handle and avoid collision with.
        """
        active_obstacles = []

        # --- Dynamic obstacle logic ---
        if self.dynamic_obstacle:
            if not self.obs_has_triggered and s >= self.obs_trigger_s:
                self.obs_has_triggered = True
                self.obs_active = True

            if self.obs_active:
                self.obs_s += self.obs_v * dt
                # Check if the obstacle has completed its path
                if self.obs_s > self.obs_end_s:
                    self.obs_active = False
                else:
                    active_obstacles.append({'s': self.obs_s, 'v': self.obs_v, 'type': 'car'})

        # --- Traffic Light logic ---
        if self.traffic_light:

            dist_2_light = self.tl_pos - s

            if self.tl_state == 'RED':

                # Check if the vehicle is close to the light position
                if 0 < dist_2_light < self.tl_trigger_s:
                    active_obstacles.append({'s': self.tl_pos, 'v': 0.0, 'type': 'light'})

                    # Check if the vehicle has stopped
                    if v < 0.1 and dist_2_light < 10.0:
                        self.tl_waiting = True

                # Countdown for switch to GREEN
                if self.tl_waiting:
                    self.tl_timer += dt
                    if self.tl_timer >= self.tl_stop_duration:
                        self.tl_state = 'GREEN'
                        self.tl_waiting = False

        return active_obstacles, self.tl_state


def run_simulation(mpc, fsm, trajectory):
    """
    Run the entire autonomous driving simulation for the reference trajectory.
    """
    # Initial state
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.5])  # Start with small speed for allow integration move forward
    current_s = x[0]

    # History storage for plots and animation
    hist_x = [x]        # States
    hist_u = []         # Controls
    hist_t = []         # CPU times
    hist_preds = []     # Predicted states
    hist_obs_s = []     # Obstacle positions
    hist_tl_state = []  # Traffic Light states

    step = 0

    while current_s <= trajectory.s_max - 1.0:

        # Update environment
        obstacles, tl_state = fsm.update(mpc.dt, x[0], x[4])

        # Solve MPC
        u_opt, pred_X, cpu_time = mpc.solve(x, obstacles)

        # Dynamics
        k_ref = trajectory.get_state(current_s)[3]
        x_dot = mpc.dynamics(x, u_opt, k_ref)
        x_next = x + mpc.dt * x_dot  # Explicit-Euler step

        # Update initial state for next iteration
        x = x_next
        current_s = x[0]

        # Store Data
        hist_x.append(x_next)
        hist_u.append(u_opt)
        hist_t.append(cpu_time)
        hist_preds.append(pred_X)
        hist_tl_state.append(tl_state)

        # Extract dynamic obstacle position for animation
        obstacle_s = next((obs['s'] for obs in obstacles if obs['type'] == 'car'), np.nan)
        hist_obs_s.append(obstacle_s)

        # Print progress in console
        if step % 50 == 0:
            vehicle_info = f"s={current_s:.1f}m, v={x[4] * 3.6:.1f}km/h"
            if fsm.traffic_light:
                dist_2_tl = fsm.tl_pos - current_s
                if dist_2_tl > mpc.obstacle_safety_distance:
                    tl_info = f"color={tl_state}, distance={dist_2_tl:.1f}m"
                else:
                    tl_info = f"color={tl_state}, distance=NaN"
            else:
                 tl_info = "NaN"
            obs_info = f"{obstacle_s - current_s:.1f}m" if not np.isnan(obstacle_s) else "NaN"
            print(f"Step {step} | {vehicle_info} | Traffic Light : {tl_info} | Distance to obstacle : {obs_info}")

        step += 1

    # Sanity checks
    trajectory_tracking_check(TrajectoryTracker(), hist_x, hist_u, hist_t,
                              hist_obs_s, hist_tl_state, fsm, trajectory.s_max)

    return np.array(hist_x), np.array(hist_u), np.array(hist_t), hist_preds, hist_obs_s, hist_tl_state, trajectory


if __name__ == "__main__":

    # Offline reference trajectory
    trajectory_file = 'trajectories/trajectory2.json'

    # Load reference trajectory framework
    trajectory_framework = TrajectoryLoader(trajectory_file)

    # Real-time tracking MPC
    tracking_MPC = TrajectoryTracker(trajectory_framework)

    # Obstacles handler
    obstacles_FSM = ObstaclesFSM(dynamic_obstacle=True, traffic_light=True)

    # Track full trajectory
    results = run_simulation(tracking_MPC, obstacles_FSM, trajectory_framework)

    # Plot results
    from visualizations import driving_animation, plot_tracking_results
    hist_x, hist_u, hist_t, hist_preds, hist_obs_s, hist_tl_state, trajectory = results
    plot_tracking_results(hist_x, hist_t, hist_obs_s, hist_tl_state,
                          trajectory, obstacles_FSM, tracking_MPC)
    driving_animation(hist_x, hist_preds, hist_obs_s, hist_tl_state,
                      trajectory, obstacles_FSM)
