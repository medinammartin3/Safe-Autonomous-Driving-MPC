import numpy as np
from path_planning import get_reference_path, get_speed_limits
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt



class TrajectoryOptimizer:
    """
    This is computed offline.
    By solving the problem, we get the desired reference path and velocity
    trajectories that will be later used in online tracking.
    """

    def __init__(self, horizon=None, N=None, dt=None, w_y=10.0, w_s=10.0, w_u=0.1, w_slack=100.0):

        # Multiple shooting configuration
        self.T = horizon
        self.N = N
        if dt is None:
            self.dt = horizon / N
        else:
            self.dt = dt


        # Weights
        self.w_y = float(w_y)
        self.w_s = float(w_s)
        self.w_u = float(w_u)
        self.w_slack = float(w_slack)

        # Control bounds
        self.u_min = np.array([-0.5, -5.0])
        self.u_max = np.array([0.5, 3.0])

        # Curvature bounds
        self.k_min = -0.5
        self.k_max = 0.5

        # Lateral acceleration bound
        self.a_n_max = 4.0


    @staticmethod
    def dynamics(x, u, k_ref):
        """
        State vector:
            x = [s, d, o, k, v], where :
            - s is the arc length of a reference point on the vehicle relative to the reference path.
                (Distance traveled / Cumulative length of the spline)
            - d is the normal distance between this point and the reference path.
            - o is the deviation in orientation with respect to the reference path.
            - k and k_ref represent the curvature of the vehicle’s trajectory and reference trajectory respectively.
            - v is the velocity of the vehicle.

        Inputs:
            u = [u1, u2], where:
            u1 : curvature rate (dk/dt)
            u2 : acceleration (dv/dt)

        Parameter:
            k_ref : curvature of reference path at the current arc length

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


    """ === OPTIMIZATION VARIABLES === """

    def unpack(self, z):
        """
        Convert the flat vector back into structured matrices.

        z structure: [ X0, X1, ..., XN, U0, ..., U_{N-1}, slack0...slack_{N-1} ]
        """
        states_vars_num = 5
        controls_num = 2
        N = self.N

        # States
        end_x_idx = (N + 1) * states_vars_num
        X = z[0 : end_x_idx]
        X = X.reshape(N + 1, states_vars_num) # Matrix od shape (N+1, 5)

        # Controls
        end_u_idx = end_x_idx + N * controls_num
        U = z[end_x_idx : end_u_idx]
        U = U.reshape(N, controls_num)

        # Slack
        S = z[end_u_idx:]
        return X, U, S

    @staticmethod
    def pack(X, U, S):
        """
        Convert the structured variables into one flat vector
        """

        flatten_X = X.ravel()  # (N+1)*5 vector
        flatten_U = U.ravel()  # N*2 vector
        flatten_S = S.ravel()  # N vector

        # Unify into a single vector
        z = np.concatenate([flatten_X, flatten_U, flatten_S])
        return z


    def cost(self, z, x0, s_final, k_ref_fun):
        """"
        Compute total cost.

        There exists a subset of the system states y = [d, o] that represent tracking errors to the reference
        trajectory, such that the path tracking problem equates to stabilizing y(t) to 0.

        y_k = [d_k, o_k] lateral + orientation deviation
        s_k = progress
        u_k = [u_1, u_2] control inputs
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

            # Evaluate reference curvature
            k_ref = k_ref_fun(float(s_k))


            # --- Cost function terms ---

            # 1. Tracking error (lateral + orientation)
            y = np.array([d_k, o_k])
            term1 = self.w_y * y @ y

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


    def constraints(self, x0, s_final, k_ref_fun, v_min_fun, v_max_fun):
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
        # (Stop at the end: s = s_final, v = 0)
        def terminal_s(z):
            X, _, _ = self.unpack(z)
            s_N = X[N, 0]
            return s_N - s_final
        constraints.append({'type': 'eq', 'fun': terminal_s})

        def terminal_v(z):
            X, _, _ = self.unpack(z)
            v_N = X[N, 4]
            return v_N
        constraints.append({'type': 'eq', 'fun': terminal_v})


        # ---- Safety constraints ----
        # (Velocity bounds : v_min <= v + slack_v <= v_max)
        for k in range(N + 1):
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

            # (Lateral acceleration : avoid excessive lateral forces)
            # -a_n_max <= k * v ^ 2 <= a_n_max
            def lateral_max(z, k=k):
                X, _, _ = self.unpack(z)
                k_k = X[k, 3]
                v_k = X[k, 4]
                return self.a_n_max - (k_k * v_k ** 2)
            constraints.append({"type": "ineq", "fun": lateral_max})

            def lateral_min(z, k=k):
                X, _, _ = self.unpack(z)
                k_k = X[k, 3]
                v_k = X[k, 4]
                return self.a_n_max + (k_k * v_k ** 2)
            constraints.append({"type": "ineq", "fun": lateral_min})


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




    def optimize(self, x0, s_final, k_ref_fun, v_min_fun, v_max_fun, min_speed_limit):
        """
        Solve the NLP
        """
        N = self.N

        # --- Initial guess (Warm start) ---
        # Linearly interpolate s from 0 to s_final and assume a constant moderate velocity.

        # States
        X_init = np.zeros((N + 1, 5))
        X_init[:, 0] = np.linspace(x0[0], s_final, N + 1)  # Linearly increase s
        X_init[:, 4] = min_speed_limit  # Set velocity to the min speed limit on the path
        X_init[-1, 4] = 0.0  # Set final velocity to 0

        # Controls
        U_init = np.zeros((N, 2))

        # Slack
        S_init = np.zeros(N)

        z0 = self.pack(X_init, U_init, S_init)

        constraints = self.constraints(x0, s_final, k_ref_fun, v_min_fun, v_max_fun)

        # minimize
        solution = minimize(
            fun=lambda z: self.cost(z, x0, s_final, k_ref_fun),
            x0=z0,
            method='SLSQP',  # Try trust-constr
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-4, 'disp': True}
        )

        X_sol, U_sol, S_sol = self.unpack(solution.x)
        return X_sol, U_sol, S_sol



def optimize_full_trajectory(locations):
    # Reference path
    way_points, extra_way_points, reference_path_spline = get_reference_path(locations)
    points = np.array(extra_way_points)

    # compute cumulative arc-length (distance traveled) along the spline
    distances = np.sqrt(np.diff(points[:, 0]) ** 2 + np.diff(points[:, 1]) ** 2)
    s_values = np.concatenate([[0], np.cumsum(distances)])
    s_total = s_values[-1]
    print(f"Total Reference Distance: {s_total} m")


    # --- Curvature from the spline at any s ---

    # Parametric parameter
    t_max = len(points) - 1
    t_values = np.linspace(0.0, t_max, len(points))
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

    speed_limits = get_speed_limits(locations, way_points, extra_way_points)

    # Max speed
    v_max_array = np.ones(len(extra_way_points))
    for limit in speed_limits:
        (_, _), (idx_start, idx_end), value = limit
        v_max_array[idx_start : idx_end + 1] = value  # assign speed value to that path segment

    min_speed_limit = np.min(v_max_array)

    # Interpolation because data is discrete but physics is continuous
    v_max_interpolator = interp1d(s_values, v_max_array, kind='previous', fill_value="extrapolate")

    def v_max_fun(s):
        return float(v_max_interpolator(s))

    # Min speed
    def v_min_fun(s):
        return 0


    # --- Run the optimization ---

    avg_speed = np.mean(v_max_array)
    est_time = s_total / avg_speed

    dt = 0.5
    # TODO: safety buffer, lagrangian, plots
    horizon = est_time
    N = int(np.ceil(horizon / dt))

    mpc = TrajectoryOptimizer(horizon=horizon, N=N, dt=dt)

    # Initial state [s, d, o, k, v]
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])


    X, U, S = mpc.optimize(x0, s_total, k_ref_fun, v_min_fun, v_max_fun, min_speed_limit)

    # Run sanity checks
    sanity_checks(mpc, X, U, S, s_total)

    return X, U, S


def plot_results(X, U, S):
    """
    Plot each variable of X, U, and S in separate figures.
    """

    state_names = ["s (arc length)", "d (lateral error)", "o (orientation error)",
                   "k (curvature)", "v (velocity)"]

    # State variables (X)
    for i in range(5):
        plt.figure()
        plt.plot(X[:, i], marker='o')
        plt.title(f"State variable X[{i}] - {state_names[i]}")
        plt.xlabel("Step k")
        plt.ylabel(state_names[i])
        plt.grid(True)
        plt.tight_layout()


    # Plot controls (U)
    control_names = ["u1 (curvature rate)", "u2 (acceleration)"]
    for i in range(2):
        plt.figure()
        plt.plot(U[:, i], marker='o', color='orange')
        plt.title(f"Control U[{i}] - {control_names[i]}")
        plt.xlabel("Step k")
        plt.ylabel(control_names[i])
        plt.grid(True)
        plt.tight_layout()


    # Slack (S)
    plt.figure()
    plt.plot(S, marker='o', color='green')
    plt.title("Slack variable S")
    plt.xlabel("Step k")
    plt.ylabel("slack")
    plt.grid(True)
    plt.tight_layout()

    plt.show()



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
        print(f'Full stop at the end : False --> Final velocity = {v_final} km/h')
        passed = False
    else:
        print(f'Full stop at the end : True')


    # --- Reverse Driving ---
    min_v = np.min(X[:, 4])
    if min_v < -VELOCITY_TOL:
        print(f'Always non-negative velocity : False --> Minimum velocity = {min_v} km/h')
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



if __name__ == '__main__':
    # Route locations
    start = 'Avenue Lincoln 1680, H3H 1G9 Montréal, Québec, Canada'
    end = 'Tim Hortons, Rue Guy 2081, H3H 2L9 Montréal, Québec, Canada'
    # start = 'McGill University'
    # end = 'Université de Montréal'
    # start = 'McGill University'
    # end = 'Avenue Lincoln 1680, H3H 1G9 Montréal, Québec, Canada'
    locations_list = [start, end]

    X, U, S = optimize_full_trajectory(locations_list)

    plot_results(X, U, S)



