import numpy as np
from path_planning import get_route, get_path_and_speed_limits
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# TODO: Lagrangians, Sanity checks

class TrajectoryOptimizer:
    """
    This is computed offline.
    By solving the problem, we get the optimal reference trajectory that will be used in online tracking.
    """

    def __init__(self, horizon=None, N=None, dt=None, vehicle_type='car', w_y=10.0, w_s=10.0, w_u=0.1, w_slack=100.0):

        # Multiple shooting configuration
        self.T = horizon
        self.N = N
        self.dt = dt

        # Weights
        self.w_y = float(w_y)
        self.w_s = float(w_s)
        self.w_u = float(w_u)
        self.w_slack = float(w_slack)

        # Control bounds
        """
        u1_min = Max Steering Rate (Right)
        u1_max = Min Steering Rate (Left)
        
        u2_min = Max Braking Deceleration
        u2_max = Min Engine Acceleration
        """
        if vehicle_type == 'car':
            self.u_min = np.array([-0.6, -5.0])
            self.u_max = np.array([0.6, 4.0])
        if vehicle_type == 'truck':
            self.u_min = np.array([-0.2, -4.0])
            self.u_max = np.array([0.2, 3.5])

        # Curvature bounds
        """
        k_min = Max Curvature (Right Turn)
        k_max = Max Curvature (Left Turn)
        """
        if vehicle_type == 'car':
            self.k_min = -0.8
            self.k_max = 0.8
        if vehicle_type == 'truck':
            self.k_min = -0.4
            self.k_max = 0.4

        # Lateral acceleration bound
        self.a_max = 4.0


    @staticmethod
    def dynamics(x, u, k_ref):
        """
        State vector:
            x = [s, d, o, k, v], where :
            - s : distance traveled by the vehicle.
            - d : lateral deviation (error) of the vehicle wrt the reference path.
            - o : orientation deviation (error) of the vehicle wrt the reference path.
            - k : curvature of the vehicle’s trajectory.
            - v : velocity of the vehicle.

        Inputs:
            u = [u1, u2], where:
            u1 : curvature rate (dk/dt)
            u2 : acceleration (dv/dt)

        Parameter:
            k_ref : curvature of the reference path at the current distance s.

        Return:
            x_dot = time derivative of state vector
        """
        s, d, o, k, v = x
        u1, u2 = u

        denom = 1 - d * k_ref
        # Avoid division by zero
        if abs(denom) < 1e-4:
            denom = 1e-4 * np.sign(denom)

        # Dynamics
        s_dot = (v * np.cos(o)) / denom
        d_dot = v * np.sin(o)
        o_dot = v * k - s_dot * k_ref
        k_dot = u1
        v_dot = u2

        # Derivative
        x_dot = np.array([s_dot, d_dot, o_dot, k_dot, v_dot])

        return x_dot


    def integrate(self, x, u, k_ref_fun):
        """
        RK4 Integration.
        Compute the next state: x_k+1 = x_k + dt * f(x,u)
        We evaluate k_ref at the specific s for each RK stage.
        """
        dt = self.dt       # Time step
        f = self.dynamics  # Dynamics

        # Left endpoint
        k_ref_1 = k_ref_fun(x[0])
        k1 = f(x, u, k_ref_1)

        # Midpoint 1
        x2 = x + 0.5 * dt * k1
        k_ref_2 = k_ref_fun(x2[0])
        k2 = f(x2, u, k_ref_2)

        # Midpoint 2
        x3 = x + 0.5 * dt * k2
        k_ref_3 = k_ref_fun(x3[0])
        k3 = f(x3, u, k_ref_3)

        # Right endpoint
        x4 = x + dt * k3
        k_ref_4 = k_ref_fun(x4[0])
        k4 = f(x4, u, k_ref_4)

        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next


    def unpack(self, z):
        """
        Convert the flat vector z into structured matrices.

        z structure: [ X0, X1, ..., XN, U0, ..., U_{N-1}, slack0...slack_{N-1} ]
        """
        states_vars_num = 5
        controls_num = 2
        N = self.N

        # States
        end_x_idx = (N + 1) * states_vars_num
        X = z[0 : end_x_idx]
        X = X.reshape(N + 1, states_vars_num)  # Matrix of shape (N+1, 5)

        # Controls
        end_u_idx = end_x_idx + N * controls_num
        U = z[end_x_idx : end_u_idx]
        U = U.reshape(N, controls_num)  # Matrix of shape (N, 2)

        # Slack
        S = z[end_u_idx:]  # N vector
        return X, U, S


    @staticmethod
    def pack(X, U, S):
        """
        Convert the structured matrices into one flat vector z.
        """
        flatten_X = X.ravel()  # (N+1)*5 vector
        flatten_U = U.ravel()  # N*2 vector
        flatten_S = S.ravel()  # N vector

        # Unify into a single vector
        z = np.concatenate([flatten_X, flatten_U, flatten_S])
        return z


    def cost(self, z, x0, s_final):
        """"
        Compute total cost.

        There exists a subset of the system states y = [d, o] that represent tracking errors to the reference path,
        such that the path tracking problem equates to stabilizing y(t) to 0.

        y_k = [d_k, o_k] --> lateral + orientation deviation
        s_k = progress  --> distance traveled by the vehicle so far
        u_k = [u_1, u_2] --> control inputs
        """
        X, U, S = self.unpack(z)

        cost = 0.0

        # For each window
        for k in range(self.N):
            # Extract values
            x_k = X[k]
            u_k = U[k]
            slack_k = S[k]

            s_k, d_k, o_k, k_k, v_k = x_k

            # --- Cost function terms ---

            # 1. Tracking error (lateral + orientation)
            y_k = np.array([d_k, o_k])
            term1 = self.w_y * y_k @ y_k

            # 2. Progress cost (Minimize distance to target)
            s0 = x0[0]
            denom = max(1, s_final - s0)  # No normalization if path length < 1 meter
            term2 = self.w_s * ((s_final - s_k) / denom) ** 2

            # 3. Control effort (Reduce energy consumption and maximize passenger comfort)
            term3 = self.w_u * np.dot(u_k, u_k)

            # 4. Slack penalty (Minimizes the slack variable for velocity relaxation)
            term4 = self.w_slack * (slack_k ** 2)

            cost += term1 + term2 + term3 + term4
        return cost


    def constraints(self, x0, s_final, k_ref_fun, v_min_fun, v_max_fun, is_final_chunk=False):
        """
        System constraints.
        """
        N = self.N
        constraints = []

        # --- Dynamics constraints ---
        # X[k+1] = RK4(X[k], U[k])
        # (Enforce continuity between Multiple Shooting windows)
        for k in range(N):
            def dynamics_constraints(z, k=k):
                X, U, _ = self.unpack(z)
                return X[k + 1] - self.integrate(X[k], U[k], k_ref_fun)
            constraints.append({
                "type": "eq",
                "fun": dynamics_constraints
            })


        # --- Initial state constraint ---
        def initial_x0(z):
            X, _, _ = self.unpack(z)
            return X[0] - x0
        constraints.append({"type": "eq", "fun": initial_x0})


        # --- Terminal constraints ----
        # (s = s_final)
        def terminal_s(z):
            X, _, _ = self.unpack(z)
            s_N = X[N, 0]
            return s_N - s_final
        constraints.append({'type': 'eq', 'fun': terminal_s})

        # v = 0 only at the final horizon
        if is_final_chunk:
            def terminal_v(z):
                X, _, _ = self.unpack(z)
                v_N = X[N, 4]
                return v_N
            constraints.append({'type': 'eq', 'fun': terminal_v})


        # ---- Safety constraints ----
        for k in range(N + 1):
            # (Velocity bounds : v_min <= v + slack_v <= v_max)
            def speed_min(z, k=k):
                X, _, S = self.unpack(z)
                if k < N:
                    slack = S[k]
                else:
                    slack = 0 # No slack on final state
                s_k = X[k, 0]
                v_k = X[k, 4]
                return (v_k + slack) - v_min_fun(s_k)
            constraints.append({"type": "ineq", "fun": speed_min})

            def speed_max(z, k=k):
                X, _, S = self.unpack(z)
                if k < N:
                    slack = S[k]
                else:
                    slack = 0
                s_k = X[k, 0]
                v_k = X[k, 4]
                return v_max_fun(s_k) - (v_k + slack)
            constraints.append({"type": "ineq", "fun": speed_max})

            # (Lateral acceleration : avoid excessive lateral forces)
            # -a_max <= k * v ^ 2 <= a_max
            def lateral_max(z, k=k):
                X, _, _ = self.unpack(z)
                k_k = X[k, 3]
                v_k = X[k, 4]
                return self.a_max - (k_k * v_k ** 2)

            constraints.append({"type": "ineq", "fun": lateral_max})

            def lateral_min(z, k=k):
                X, _, _ = self.unpack(z)
                k_k = X[k, 3]
                v_k = X[k, 4]
                return self.a_max + (k_k * v_k ** 2)

            constraints.append({"type": "ineq", "fun": lateral_min})


        # --- Physical constraints ---
        # (Curvature limits)
        for k in range(N + 1):
            def k_min(z, k=k):
                X, _, _ = self.unpack(z)
                k_k = X[k, 3]
                return k_k - self.k_min
            constraints.append({"type": "ineq", "fun": k_min})

            def k_max(z, k=k):
                X, _, _ = self.unpack(z)
                k_k = X[k, 3]
                return self.k_max - k_k
            constraints.append({"type": "ineq", "fun": k_max})




        # --- Controls constraints ---
        # (Curvature bounds)
        for k in range(N):
            def u1_min(z, k=k):
                _, U, _ = self.unpack(z)
                u1_k = U[k, 0]
                return u1_k - self.u_min[0]
            constraints.append({"type": "ineq", "fun": u1_min})

            def u1_max(z, k=k):
                _, U, _ = self.unpack(z)
                u1_k = U[k, 0]
                return self.u_max[0] - u1_k
            constraints.append({"type": "ineq", "fun": u1_max})

            # (Acceleration/deceleration bounds)
            def u2_min(z, k=k):
                _, U, _ = self.unpack(z)
                u2_k = U[k, 1]
                return u2_k - self.u_min[1]
            constraints.append({"type": "ineq", "fun": u2_min})

            def u2_max(z, k=k):
                _, U, _ = self.unpack(z)
                u2_k = U[k, 1]
                return self.u_max[1] - u2_k
            constraints.append({"type": "ineq", "fun": u2_max})

            # (Slack non-negative)
            def non_negative_slack(z, k=k):
                _, _, S = self.unpack(z)
                return S[k]
            constraints.append({"type": "ineq", "fun": non_negative_slack})

        return constraints




    def optimize(self, x0, s_final, k_ref_fun, v_min_fun, v_max_fun, min_speed_limit, is_final_chunk=False):
        """
        Solve the NLP
        """
        N = self.N

        # --- Initial guess (Warm start) ---
        # Linearly interpolate s from 0 to s_final and assume a constant moderate velocity.

        # State
        X_init = np.zeros((N + 1, 5))
        X_init[:, 0] = np.linspace(x0[0], s_final, N + 1)  # Linearly increase s
        if is_final_chunk:
            # Slow down velocity until 0 on final horizon
            X_init[:, 4] = np.linspace(max(x0[4], min_speed_limit), 0.0, N + 1)
        else:
            # Keep constant speed guess on intermediate horizons
            X_init[:, 4] = max(x0[4], min_speed_limit)

        # Controls
        U_init = np.zeros((N, 2))

        # Slack
        S_init = np.zeros(N)

        z0 = self.pack(X_init, U_init, S_init)

        constraints = self.constraints(x0, s_final, k_ref_fun, v_min_fun, v_max_fun, is_final_chunk)

        # minimize
        solution = minimize(
            fun=lambda z: self.cost(z, x0, s_final),
            x0=z0,
            method='SLSQP',  # Try trust-constr
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-4, 'disp': True}
        )

        X_sol, U_sol, S_sol = self.unpack(solution.x)
        return X_sol, U_sol, S_sol


def unpack_reference_path(route):
    """
    Get the reference path from GraphHopper and calculate the cumulative distance s and total path length.

    Parameters:
        - route : route from GraphHopper.

    Returns:
        - points : list of detailed way-points in local coordinates.
        - spline : cubic spline representing the reference path.
        - speed_limits: list of speed limit intervals and values.
        - s_values : cumulative distance traveled at each point.
        - s_total : total length of the reference path.
    """
    # Get reference path
    (_, detailed_points, spline), speed_limits = get_path_and_speed_limits(route)
    points = np.array(detailed_points)

    # Calculate total distance
    distances = np.sqrt(np.diff(points[:, 0]) ** 2 + np.diff(points[:, 1]) ** 2)
    s_values = np.concatenate([[0], np.cumsum(distances)])
    s_total = s_values[-1]

    return  [points, spline, speed_limits, s_values, s_total]


def optimize_full_trajectory(route, max_chunk_size=100):
    """
    Compute the full optimal reference trajectory with a chunked optimization scheme / receding horizon.
    """
    # Reference path
    detailed_points, reference_path_spline, speed_limits, s_values, s_total = unpack_reference_path(route)

    # --- Curvature from the spline at any s ---

    # Parametric parameter
    t_max = len(detailed_points) - 1
    t_values = np.linspace(0.0, t_max, len(detailed_points))
    s_to_t = interp1d(s_values, t_values, kind='linear', fill_value='extrapolate')

    # Curvature
    def k_ref_fun(s):
        t = float(s_to_t(s))
        x_spline, y_spline = reference_path_spline

        # Derivatives wrt t
        x_dt = x_spline(t, 1)
        y_dt = y_spline(t, 1)
        x_ddt = x_spline(t, 2)
        y_ddt = y_spline(t, 2)

        # Curvature formula
        denom = (x_dt ** 2 + y_dt ** 2) ** 1.5 + 1e-9
        if denom < 1e-8: denom = 1e-8  # Avoid division by 0
        k = (x_dt * y_ddt - y_dt * x_ddt) / denom
        return float(k)

    # --- Velocity limits ---

    # Max speed
    v_max_array = np.ones(len(detailed_points))
    for limit in speed_limits:
        (_, _), (idx_start, idx_end), value = limit
        v_max_array[idx_start : idx_end + 1] = value  # assign speed limit to that path segment

    min_speed_limit = np.min(v_max_array)

    # Interpolation because data is discrete but physics is continuous
    v_max_interpolator = interp1d(s_values, v_max_array, kind='previous', fill_value="extrapolate")

    def v_max_fun(s):
        return float(v_max_interpolator(s))

    # Min speed
    def v_min_fun(s):
        return 0

    # --- Run the optimization ---

    X_full = []
    U_full = []
    S_full = []

    # Initial state for chunk 1
    current_x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    print(f"Total Distance: {s_total} m")
    remaining_dist = s_total
    chunk_nb = 1

    while remaining_dist > 0.1:

        chunk_size = min(max_chunk_size, remaining_dist) # meters

        current_s = current_x0[0]
        s_target = current_s + chunk_size

        print(f"==> Chunk #{chunk_nb} : {current_s} --> {s_target}")

        is_final_chunk = (abs(s_target - s_total) < 1.0)

        avg_speed = np.mean(v_max_array[int(current_s/5):]) # Points every 5 m
        est_time = chunk_size / avg_speed

        # Safety buffer : give enough time to make full path (avoid Infeasibility)
        # 20% if intermediate, 50% if final (vehicle needs to slow down to stop)
        safety_buffer = 1.5 if is_final_chunk else 1.2

        horizon = est_time * safety_buffer  # seconds
        dt = 0.5  # seconds
        N = int(np.ceil(horizon / dt))  # seconds

        mpc = TrajectoryOptimizer(horizon=horizon, N=N, dt=dt)

        X, U, S = mpc.optimize(current_x0, s_target, k_ref_fun, v_min_fun, v_max_fun, min_speed_limit)

        # Solution concatenation
        # --> len(X) = N+1
        # --> Last point of Chunk K = start point of Chunk K+1.
        if len(X_full) == 0:
            X_full.append(X)
            U_full.append(U)
            S_full.append(S)
        else:
            # Avoid duplicated first point
            X_full.append(X[1:])
            U_full.append(U)
            S_full.append(S)

        # Update state for next loop
        current_x0 = X[-1]  # Start next chunk where we ended
        remaining_dist = s_total - current_x0[0]

        chunk_nb += 1

    # Concatenate chunk solutions
    X_final = np.concatenate(X_full, axis=0)
    U_final = np.concatenate(U_full, axis=0)
    S_final = np.concatenate(S_full, axis=0)

    # Sanity Check on the full result
    sanity_checks(TrajectoryOptimizer(), X_final, U_final, S_final, s_total)

    return X_final, U_final, S_final


def sanity_checks(mpc, X, U, S, s_total):
    """
    Verifies the physical feasibility of the optimized trajectory.
    """
    print("\n=== SANITY CHECKS ===")

    passed = True

    # Thresholds
    DISTANCE_TOL = 0.5  # meters
    VELOCITY_TOL = 0.1  # km/h
    CONTROLS_TOL = 0.1
    LATERAL_DEV_TOL = 4.0  # meters (Max lateral deviation allowed)
    SLACK_TOL = 0.1

    # --- Destination ---
    s_final = X[-1, 0]
    error_s = abs(s_final - s_total)
    if error_s > DISTANCE_TOL:
        print(f'Final destination reached : False --> Error = {error_s} m')
        passed = False
    else:
        print(f'Final destination reached : True')

    # --- Full Stop ---
    v_final = X[-1, 4]
    if abs(v_final) > VELOCITY_TOL:
        print(f'Full stop at the end : False --> Final velocity = {v_final*3.6} km/h')
        passed = False
    else:
        print(f'Full stop at the end : True')

    # --- Reverse Driving ---
    min_v = np.min(X[:, 4])
    if min_v < -VELOCITY_TOL:
        print(f'Always non-negative velocity : False --> Minimum velocity = {min_v*3.6} km/h')
        passed = False
    else:
        print(f'Always non-negative velocity : True')

    # --- Control Bounds ---
    u1_min, u2_min = mpc.u_min
    u1_max, u2_max = mpc.u_max

    # U1 (Curvature)
    u1_check = not(np.min(U[:, 0]) < u1_min - CONTROLS_TOL and np.max(U[:, 0]) > u1_max + CONTROLS_TOL)
    print(f'Curvature rate limits respected : {u1_check}')
    passed = u1_check

    # U2 (Acceleration)
    u2_check = not(np.min(U[:, 1]) < u2_min - CONTROLS_TOL and np.max(U[:, 1]) > u2_max + CONTROLS_TOL)
    print(f'Acceleration limits respected : {u2_check}')
    passed = u2_check

    # --- Lateral Deviation ---
    max_lat_dev = np.max(np.abs(X[:, 1]))
    if max_lat_dev > LATERAL_DEV_TOL:
        print(f'Lateral deviation respected : False --> Maximum lateral deviation : {max_lat_dev} m')
        passed = False
    else:
        print(f'Lateral deviation respected : True')

    # --- Slack Variable ---
    max_slack = np.max(np.abs(S))
    if max_slack > SLACK_TOL:
        print(f'Low slack usage : False --> Maximum slack value : {max_slack}')
        passed = False
    else:
        print(f'Low slack usage : True')

    print(f'===> Checks passed : {passed}')


def plot_results(X, U, S):
    """
    Plot each variable of X, U, and S in separate figures.
    """
    marker = 'o'
    marker_size = 3

    # --- State variables (X) ---
    state_names = [
        "Distance Traveled (s)",
        "Lateral Deviation (d)",
        "Orientation Deviation (o)",
        "Curvature (k)",
        "Velocity (v)"
    ]

    # Travel distance + Curvature + Velocity
    phys_indices = [0, 3, 4]
    fig_phys, axs_phys = plt.subplots(1, 3, figsize=(18, 5))
    fig_phys.suptitle("Optimal Vehicle State Trajectory (Physics)", fontsize=14)
    for i, x_idx in enumerate(phys_indices):
        ax = axs_phys[i]
        if x_idx == 0:  # Distance
            ax.plot(X[:, x_idx], marker=marker, ms=marker_size, color='tab:blue')
            ax.set_ylabel("Distance ($m$)")
        if x_idx == 3:  # Curvature
            ax.plot(X[:, x_idx], marker=marker, ms=marker_size, color='tab:blue')
            ax.set_ylabel(r"Curvature ($m^{-1}$)")
        if x_idx == 4:  # Velocity
            ax.plot(X[:, x_idx] * 3.6, marker=marker, ms=marker_size, color='tab:blue')  # m/s to km/h
            ax.set_ylabel("Velocity ($km/h$)")
        ax.set_title(state_names[x_idx])
        ax.set_xlabel("Step")
        ax.grid(True)
    plt.tight_layout(w_pad=1.7)

    # Lateral deviation + Orientation deviation
    err_indices = [1, 2]
    fig_err, axs_err = plt.subplots(1, 2, figsize=(12, 5))
    fig_err.suptitle("Trajectory Tracking Errors", fontsize=14)
    for i, x_idx in enumerate(err_indices):
        ax = axs_err[i]
        if x_idx == 1:  # Lateral
            ax.plot(X[:, x_idx], marker=marker, ms=marker_size, color='tab:red')
            ax.set_ylabel(r"Deviation ($m$)")
        if x_idx == 2:  # Orientation
            ax.plot(X[:, x_idx], marker=marker, ms=marker_size, color='tab:red')
            ax.set_ylabel(r"Deviation ($rad$)")
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label="Ideal")  # Zero line (ideal value)
        ax.set_title(state_names[x_idx])
        ax.set_xlabel("Step")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()


    # --- Controls (U) ---
    control_names = ["Curvature Rate ($u_1$)", "Acceleration ($u_2$)"]

    fig_U, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig_U.suptitle("Optimal Controls (U)", fontsize=14)

    for i in range(2):
        if i == 0:  # Curvature
            axs[i].plot(U[:, i], marker=marker, color='tab:orange', ms=marker_size)
            axs[i].set_ylabel(r"Input Value ($m^{-1}s^{-1}$)")
        if i == 1:  # Acceleration
            axs[i].plot(U[:, i], marker=marker, color='tab:orange', ms=marker_size)
            axs[i].set_ylabel(r"Input Value ($m/s^2$)")
        axs[i].plot(U[:, i], marker=marker, color='orange', ms=marker_size)
        axs[i].set_title(control_names[i])
        axs[i].set_xlabel("Step")
        axs[i].grid(True)
    plt.tight_layout()


    # --- Slack (S) ---
    plt.figure()
    plt.plot(S * 3.6, marker=marker, color='tab:green', ms=marker_size)  # m/s to km/h
    plt.title("Optimal Velocity Slack Variable (S)")
    plt.xlabel("Step")
    plt.ylabel(r"Value ($km/h$)")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


def recreate_trajectory(X, route):
    """
    Reconstructs and plots the path of the optimized trajectory (tracked from the GraphHopper reference path).
    It also plots the reference path and add visual road boundaries to verify the vehicle's tracking performance
    and path feasibility.
    """
    # Reference path
    detailed_points, reference_path_spline, speed_limits, ref_s_values, s_total = unpack_reference_path(route)

    # Total trajectory time
    dt = 0.5
    total_time = (len(X) - 1) * dt

    # Interpolator s -> t (parametric)
    t_max = len(detailed_points) - 1
    t_values = np.linspace(0.0, t_max, len(detailed_points))
    s_to_t = interp1d(ref_s_values, t_values, kind='linear', fill_value='extrapolate')

    x_spl, y_spl = reference_path_spline

    # Transform MPC solution (s, d) -> (x, y)
    traj_x = []
    traj_y = []

    # Road boundaries
    left_bound_x, left_bound_y = [], []
    right_bound_x, right_bound_y = [], []

    lane_width = 3.5  # meters

    # --- Re-create the reference path (spline) and MPC trajectory (optimal solution) ---
    for i in range(len(X)):
        s_vehicle = X[i, 0]
        d_vehicle = X[i, 1]

        # Reference point
        t = float(s_to_t(s_vehicle))
        x_ref = float(x_spl(t))
        y_ref = float(y_spl(t))

        # Reference angle (tangent angle of the spline)
        dx = float(x_spl(t, 1))
        dy = float(y_spl(t, 1))
        angle = np.arctan2(dy, dx)

        # Calculate vehicle's (MPC trajectory) position
        # d > 0 = Left, d < 0 = Right
        x_actual = x_ref - d_vehicle * np.sin(angle)
        y_actual = y_ref + d_vehicle * np.cos(angle)

        traj_x.append(x_actual)
        traj_y.append(y_actual)

        # Calculate road boundaries
        left_bound_x.append(x_ref - lane_width * np.sin(angle))
        left_bound_y.append(y_ref + lane_width * np.cos(angle))
        right_bound_x.append(x_ref - (-lane_width) * np.sin(angle))
        right_bound_y.append(y_ref + (-lane_width) * np.cos(angle))

    # --- Plot ---
    plt.figure(figsize=(12, 8))

    # Plot Reference Path (centerline)
    fine_t = np.linspace(0, t_max, 1000)
    plt.plot(x_spl(fine_t), y_spl(fine_t), 'k--', alpha=0.5, label='Reference Path (GraphHopper)')

    # Plot road boundaries
    plt.plot(left_bound_x, left_bound_y, 'k-', linewidth=0.5, alpha=0.3)
    plt.plot(right_bound_x, right_bound_y, 'k-', linewidth=0.5, alpha=0.3, label='Road Boundaries')

    # Plot MPC Trajectory
    velocities = X[:, 4] * 3.6  # m/s to km/h
    trajectory = plt.scatter(traj_x, traj_y, c=velocities, cmap='plasma', s=30, label='Optimized Trajectory (MPC)')

    plt.colorbar(trajectory, label='Velocity (km/h)')
    plt.legend()
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Optimal Trajectory Visualization | Distance : {s_total:.1f} m | Time : {total_time:.1f} s')
    plt.axis('equal')
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    # Route locations
    start = 'Avenue Lincoln 1680, H3H 1G9 Montréal, Québec, Canada'
    end = 'Tim Hortons, Rue Guy 2081, H3H 2L9 Montréal, Québec, Canada'
    # start = 'McGill University'
    # end = 'Université de Montréal'
    # start = 'McGill University'
    # end = 'Avenue Lincoln 1680, H3H 1G9 Montréal, Québec, Canada'
    locations_list = [start, end]
    vehicle = 'car'  # ['car', 'truck']

    # Route
    GraphHopper_route = get_route(locations_list, vehicle)

    # Trajectory
    X, U, S = optimize_full_trajectory(GraphHopper_route)

    # Plot solution
    # plot_results(X, U, S)

    # Plot trajectory visualization
    recreate_trajectory(X, GraphHopper_route)



