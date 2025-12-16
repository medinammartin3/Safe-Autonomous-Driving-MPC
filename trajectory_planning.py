import numpy as np
from sanity_checks import reference_trajectory_check
from path_planning import get_route, get_path_and_speed_limits
from scipy.optimize import minimize
from scipy.interpolate import interp1d


class TrajectoryOptimizer:
    """
    This is computed offline.
    By solving the problem, we get the optimal reference trajectory that will be used in online tracking.
    """

    def __init__(self, horizon=None, N=None, dt=None, w_y=10.0, w_s=10.0, w_u=0.1, w_slack=100.0):

        # Configuration
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
        self.u_min = np.array([-0.6, -5.0])
        self.u_max = np.array([0.6, 4.0])

        # Curvature bounds
        """
        k_min = Max Curvature (Right Turn)
        k_max = Max Curvature (Left Turn)
        """
        self.k_min = -0.8
        self.k_max = 0.8

        # Lateral acceleration bound
        self.a_max = 6.0

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
            denom = 1e-4 * np.sign(denom) if denom != 0 else 1e-4

        # Dynamics
        s_dot = (v * np.cos(o)) / denom
        d_dot = v * np.sin(o)
        o_dot = v * k - s_dot * k_ref
        k_dot = u1
        v_dot = u2

        # Derivative
        x_dot = np.array([s_dot, d_dot, o_dot, k_dot, v_dot])

        return x_dot

    def unpack(self, z):
        """
        Convert the flat vector z into structured matrices.

        z structure: [ X0, X1, ..., XN, U0, ..., U_{N-1}, S_0...S{N-1} ]
        """
        states_vars_num = 5
        controls_num = 2
        N = self.N

        # States
        end_x_idx = (N + 1) * states_vars_num
        X = z[0: end_x_idx]
        X = X.reshape(N + 1, states_vars_num)  # Matrix of shape (N+1, 5)

        # Controls
        end_u_idx = end_x_idx + N * controls_num
        U = z[end_x_idx: end_u_idx]
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

    def cost(self, z, x0, s_total):
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

            # Tracking error (lateral + orientation)
            y_k = np.array([d_k, o_k])
            term1 = self.w_y * (y_k @ y_k)

            # Progress cost (Minimize distance to global destination/target)
            s0 = x0[0]
            denom = max(1, s_total - s0)  # No normalization if path length < 1 meter
            term2 = self.w_s * ((s_total - s_k) / denom) ** 2

            # Control effort (Reduce energy/fuel consumption and maximize passenger comfort)
            term3 = self.w_u * (u_k @ u_k)

            # Slack penalty (Minimize the slack variable for velocity relaxation)
            term4 = self.w_slack * (slack_k ** 2)

            cost += term1 + term2 + term3 + term4
        return cost

    def constraints(self, x0, s_target, k_ref_fun, v_min_fun, v_max_fun, is_final_chunk):
        """
        System constraints.
        """
        N = self.N
        constraints = []

        # --- Dynamics constraint ---
        # Enforce dynamic feasibility and continuity (defects = 0)
        for k in range(N):
            # Hermite-Simpson collocation
            def dynamics_constraints(z, k=k):
                X, U, _ = self.unpack(z)
                x_k = X[k]
                x_next = X[k + 1]
                u_k = U[k]
                dt = self.dt

                # Dynamics at the endpoints (k and k+1)
                k_ref_k = k_ref_fun(x_k[0])
                f_k = self.dynamics(x_k, u_k, k_ref_k)

                k_ref_next = k_ref_fun(x_next[0])
                f_next = self.dynamics(x_next, u_k, k_ref_next)

                # Estimate the state at the midpoint (k + 1/2)
                x_mid = 0.5 * (x_k + x_next) + (dt / 8.0) * (f_k - f_next)

                # Dynamics at the midpoint
                k_ref_mid = k_ref_fun(x_mid[0])
                f_mid = self.dynamics(x_mid, u_k, k_ref_mid)

                # Simpson's Rule
                x_pred = x_k - (dt / 6.0) * (f_k + 4 * f_mid + f_next)

                defect = x_next - x_pred
                return defect

            constraints.append({"type": "eq", "fun": dynamics_constraints})

        # --- Initial state constraint ---
        # Enforce continuity between chunks
        def initial_x0(z):
            X, _, _ = self.unpack(z)
            return X[0] - x0  # x0 = Final state of previous chunk

        constraints.append({"type": "eq", "fun": initial_x0})

        # --- Terminal constraints ----
        if is_final_chunk:
            # s_N = s_target (reach final destination)
            def terminal_s(z):
                X, _, _ = self.unpack(z)
                s_N = X[N, 0]
                return s_N - s_target

            constraints.append({'type': 'eq', 'fun': terminal_s})

            # v = 0 (stop at final destination)
            def terminal_v(z):
                X, _, _ = self.unpack(z)
                v_N = X[N, 4]
                return v_N

            constraints.append({'type': 'eq', 'fun': terminal_v})

        # s_N >= s_target/2 for intermediate chunks
        # (arrive at least to half of the chunk target and go further if you have time)
        else:
            def terminal_s(z):
                X, _, _ = self.unpack(z)
                s_N = X[N, 0]
                return s_N - s_target / 2

            constraints.append({'type': 'ineq', 'fun': terminal_s})

        # ---- Safety constraints ----
        for k in range(N + 1):
            # Velocity bounds : v_min <= v + slack_v <= v_max
            def speed_min(z, k=k):
                X, _, S = self.unpack(z)
                if k < N:
                    slack = S[k]
                else:
                    slack = 0  # No slack on final state
                s_k = X[k, 0]
                v_k = X[k, 4]
                return (v_k + slack) - v_min_fun(s_k)

            constraints.append({"type": "ineq", "fun": speed_min})

            def speed_max(z, k=k):
                X, _, S = self.unpack(z)
                if k < N:
                    slack = S[k]
                else:
                    slack = 0  # No slack on final state
                s_k = X[k, 0]
                v_k = X[k, 4]
                return v_max_fun(s_k) - (v_k + slack)

            constraints.append({"type": "ineq", "fun": speed_max})

            # Lateral acceleration : avoid excessive lateral forces
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
        # Curvature limits
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
        # Curvature bounds
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

            # Acceleration/deceleration bounds
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

            # Slack non-negative
            def non_negative_slack(z, k=k):
                _, _, S = self.unpack(z)
                return S[k]

            constraints.append({"type": "ineq", "fun": non_negative_slack})

        return constraints

    def optimize(self, x0, s_target, s_total, k_ref_fun, v_min_fun, v_max_fun, is_final_chunk):
        """
        Solve the NLP.
        """
        N = self.N

        # --- Initial guess ---
        # Linearly interpolate s from 0 to s_target and assume a constant moderate velocity.

        # State (Warm start with last state speed)
        X_init = np.zeros((N + 1, 5))
        X_init[:, 0] = np.linspace(x0[0], s_target, N + 1)  # Linearly increase s
        if is_final_chunk:
            # Slow down velocity until 0 on final horizon
            X_init[:, 4] = np.linspace(x0[4], 0.0, N + 1)
        else:
            # Keep constant initial speed guess on intermediate horizons
            X_init[:, 4] = x0[4]

        # Controls
        U_init = np.zeros((N, 2))

        # Slack
        S_init = np.zeros(N)

        z0 = self.pack(X_init, U_init, S_init)

        constraints = self.constraints(x0, s_target, k_ref_fun, v_min_fun, v_max_fun, is_final_chunk)

        # Solve with SLSQP
        solution = minimize(
            fun=lambda z: self.cost(z, x0, s_total),
            x0=z0,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-4, 'disp': True}
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

    return [points, spline, speed_limits, s_values, s_total]


def optimize_full_trajectory(route, max_chunk_size=20):
    """
    MPC for computing the full optimal reference trajectory with a closed-loop chunked (segmented) optimization scheme
    and applying receding horizon principle for ensuring feasibility and computational efficiency.
    Each chunk serves the role of a 'context window' (feedback) for telling to the optimizer what's coming next
    (turns, new speed limits, destination approaching, etc.), so he can prepare and ensure smooth transitions.
    There's no noise in the dynamics/controls, the optimal trajectory is deterministic.

    Parameters:
        - route : route from GraphHopper.
        - max_chunk_size : maximum chunk size in meters.

    Returns:
        - X, U, S : full optimal trajectory.
    """
    # Reference path
    detailed_points, reference_path_spline, speed_limits, s_values, s_total = unpack_reference_path(route)

    # --- Curvature from the spline at any s ---

    # Parametric parameter
    t_max = len(detailed_points) - 1
    t_values = np.linspace(0.0, t_max, len(detailed_points))
    s_to_t = interp1d(s_values, t_values, kind='linear', fill_value='extrapolate')

    # Reference curvature (curvature of the reference path spline)
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
        v_max_array[idx_start: idx_end + 1] = value  # assign speed limit to that path segment

    # Interpolation because data is discrete but physics are continuous
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

    print(f"Total Distance : {s_total:.2f} m")
    remaining_dist = s_total

    while remaining_dist > 0.1:

        # Choose context chunk size (meters)
        if remaining_dist < max_chunk_size * 2:
            chunk_size = remaining_dist
            is_final_chunk = True
        else:
            chunk_size = max_chunk_size
            is_final_chunk = False

        # Chunk target distance
        current_s = current_x0[0]
        s_target = current_s + chunk_size

        print(f"==> Distance traveled --> {current_s:.2f} m ({current_s / s_total * 100:.2f}%)")

        avg_speed = np.mean(v_max_array[int(current_s / 5):])  # Points every 5 m
        est_time = chunk_size / avg_speed

        # Safety buffer : give enough time to make full path (avoid infeasibility)
        safety_buffer = 2.0

        horizon = est_time * safety_buffer  # seconds
        dt = 0.3  # seconds
        N = int(np.ceil(horizon / dt))  # seconds

        optimizer = TrajectoryOptimizer(horizon=horizon, N=N, dt=dt)

        X, U, S = optimizer.optimize(current_x0, s_target, s_total, k_ref_fun,
                                v_min_fun,v_max_fun, is_final_chunk)

        # --- Receding horizon ---
        if not is_final_chunk:
            commit_steps = int(N/2)  # Only save half of the steps
            # Slice solution
            X_sliced = X[:commit_steps + 1]
            U_sliced = U[:commit_steps]
            S_sliced = S[:commit_steps]

            # Add solution to the solutions list
            # --> len(X) = N+1
            # --> Last state of chunk K = start point of Chunk K+1.
            if len(X_full) == 0:
                X_full.append(X_sliced)
                U_full.append(U_sliced)
                S_full.append(S_sliced)
            else:
                # Avoid duplicated first point
                X_full.append(X_sliced[1:])
                U_full.append(U_sliced)
                S_full.append(S_sliced)
        else:
            X_full.append(X[1:])
            U_full.append(U)
            S_full.append(S)

        # Update state for next optimization
        current_x0 = X_full[-1][-1]  # Start next chunk where we ended
        remaining_dist = s_total - current_x0[0]

    # Concatenate chunk solutions
    X_final = np.concatenate(X_full, axis=0)
    U_final = np.concatenate(U_full, axis=0)
    S_final = np.concatenate(S_full, axis=0)

    # Sanity check on the full result
    reference_trajectory_check(TrajectoryOptimizer(), X_final, U_final, S_final, s_total)

    return X_final, U_final, S_final


if __name__ == '__main__':
    # --- Route locations ---
    # trajectory1.json
    # start = 'Musée des Beaux-Arts de Montréal'
    # end = 'nesto mortgages-hypothèques'

    # trajectory2.json
    start = "Musée des Beaux-Arts de Montréal"
    end = 'McGill University'

    # trajectory3.json
    # start = "Avenue du Parc, H2V 2G4 Montréal, Québec, Canada"
    # end = "1505 Voie Camillien-Houde, Montréal, QC H3H 1A1"

    locations_list = [start, end]

    # Route
    GraphHopper_route = get_route(locations_list)

    # Trajectory
    X, U, S = optimize_full_trajectory(GraphHopper_route)

    # # Save optimal trajectory on a JSON file
    # trajectory = {
    #     'X': X.tolist(),
    #     'U': U.tolist(),
    #     'S': S.tolist()
    # }
    # import json
    # with open("trajectories/trajectory.json", "w") as file:
    #     json.dump(trajectory, file, indent=4)

    from visualizations import plot_trajectory_optimization_results, recreate_optimal_trajectory
    # Plot solution
    plot_trajectory_optimization_results(GraphHopper_route, X, U, S)
    # Plot trajectory visualization
    recreate_optimal_trajectory(X, GraphHopper_route)