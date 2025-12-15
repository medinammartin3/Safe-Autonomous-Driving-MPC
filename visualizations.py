from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from path_planning import get_speed_limits
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import matplotlib
try:
    matplotlib.use('MacOSX')
except:
    matplotlib.use('TkAgg')


def plot_reference_path(route, original_points, spline, extra_points, show_extra_points=False, show_speed_limits=False):
    """
    Plot the reference path with original points and optional extra points + annotated speed limits along the spline.
    """
    # Route data
    start_name = route['start'].split(',')[0]
    end_name = route['end'].split(',')[0]
    distance = route['distance']
    time = route['time']
    speed_limits = get_speed_limits(route)

    # Reference path spline
    spline_x, spline_y = spline

    # Parametric parameter
    t_min = 0
    t_max = len(spline_x.x) - 1
    num_samples = len(extra_points) * 50
    t_new = np.linspace(t_min, t_max, num_samples)

    # Spline coordinates
    xs_spline = spline_x(t_new)
    ys_spline = spline_y(t_new)

    # Original points
    start_x, start_y = original_points[0]
    end_x, end_y = original_points[-1]
    xs = [p[0] for p in original_points[1:-1]]
    ys = [p[1] for p in original_points[1:-1]]

    # Added extra points
    xs_extra = None
    ys_extra = None
    if show_extra_points:
        xs_extra = [p[0] for p in extra_points]
        ys_extra = [p[1] for p in extra_points]


    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.gca().set_axisbelow(True)

    # Plot spline
    plt.plot(xs_spline, ys_spline, color='peru', linestyle='-', label='Spline path', zorder=1)

    # Plot extra points
    if show_extra_points:
        plt.scatter(xs_extra, ys_extra, s=8, color='green', label='Extra way-points added', zorder=2)

    # Plot original points
    plt.scatter(xs, ys, s=20, color='blue', label='Original way-points', zorder=3)

    # Plot start/end points
    plt.scatter(start_x, start_y, s=100, color='green', label=f'{start_name} (start)', zorder=4)
    plt.scatter(end_x, end_y, s=100, color='red', label=f'{end_name} (end)', zorder=4)

    # Annotate speed limits next to the original points
    if show_speed_limits:
        for idx, (x, y) in enumerate(original_points[1:-1]):
            # Find speed limit interval
            limit_value = None
            for (interval_start, interval_end), (_, _), value in speed_limits:
                if interval_start <= idx < interval_end:
                    limit_value = value*3.6  # m/s to km/h
                    break

            plt.text(x, y + 2, f"{limit_value:.0f} km/h", color='black', fontsize=8, ha='center', fontweight='bold')
        speed_label = mlines.Line2D([], [], color='black', marker='$-$', linestyle='None', markersize=3, label='Speed limit')
        plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + [speed_label])

    else:
        plt.legend()

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Reference Path with Cubic Spline | Real Distance : {distance:.1f} m | Estimated Time : {time:.1f} s')
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_trajectory_optimization_results(route, X, U, S):
    """
    Plot each variable of X, U, and S from the optimized reference trajectory in separate figures.
    """
    # --- Constants ---
    dt_opt = 0.3
    times = np.arange(len(X)) * dt_opt
    s_traj = X[:, 0]

    marker = 'o'
    marker_size = 2

    # --- Speed Limits Data ---
    from trajectory_planning import unpack_reference_path
    points, _, speed_limits, s_values, _ = unpack_reference_path(route)
    speed_limits = get_speed_limits(route)
    v_max_array = np.ones(len(points))
    for limit in speed_limits:
        (_, _), (idx_start, idx_end), value = limit
        v_max_array[idx_start: idx_end + 1] = value  # value is in m/s
    # Interpolator: s (distance) -> v_limit (m/s)
    v_max_interpolator = interp1d(s_values, v_max_array, kind='previous', fill_value="extrapolate")
    # Map optimal trajectory distance --> speed limits
    s_trajectory = X[:, 0]
    v_limits_trajectory = v_max_interpolator(s_trajectory) * 3.6  # m/s to km/h

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
    fig_phys.suptitle("Optimal Trajectory", fontsize=14)
    for i, x_idx in enumerate(phys_indices):
        ax = axs_phys[i]
        if x_idx == 0:  # Distance
            ax.plot(times, X[:, x_idx], marker=marker, ms=marker_size, color='tab:blue')
            ax.set_ylabel("Distance [$m$]")
            ax.set_xlabel(r'Time [$s$]')
        if x_idx == 3:  # Curvature
            ax.plot(s_traj, X[:, x_idx], marker=marker, ms=marker_size, color='tab:blue')
            ax.set_ylabel(r"Curvature [$m^{-1}$]")
            ax.set_xlabel(r'Distance [$m$]')
        if x_idx == 4:  # Velocity
            # Speed limits
            ax.plot(s_traj, v_limits_trajectory, color='tomato', linestyle='--', alpha=0.7, label='Speed Limit')
            ax.legend()
            # Vehicle Velocity
            ax.plot(s_traj, X[:, x_idx] * 3.6, marker=marker, ms=marker_size, color='tab:blue')  # m/s to km/h
            ax.set_ylabel("Velocity [$km/h$]")
            ax.set_xlabel(r'Distance [$m$]')
        ax.set_title(state_names[x_idx])
        ax.grid(True)
    plt.tight_layout(w_pad=1.7)

    # Lateral deviation + Orientation deviation
    err_indices = [1, 2]
    fig_err, axs_err = plt.subplots(1, 2, figsize=(12, 5))
    fig_err.suptitle("Trajectory Tracking Errors", fontsize=14)
    for i, x_idx in enumerate(err_indices):
        ax = axs_err[i]
        if x_idx == 1:  # Lateral
            ax.axhline(y=1.5, color='tomato', linestyle='--', label='Road Limits')
            ax.axhline(y=-1.5, color='tomato', linestyle='--')
            ax.plot(s_traj, X[:, x_idx], marker=marker, ms=marker_size, color='tab:red')
            ax.set_ylabel(r"Deviation [$m$]")
        if x_idx == 2:  # Orientation
            ax.plot(s_traj, X[:, x_idx], marker=marker, ms=marker_size, color='tab:red')
            ax.set_ylabel(r"Deviation [$rad$]")
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label="Ideal")  # Zero line (ideal value)
        ax.set_title(state_names[x_idx])
        ax.set_xlabel(r'Distance [$m$]')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()

    # --- Controls (U) ---
    control_names = ["Curvature Rate [$u_1$]", "Acceleration [$u_2$]"]

    fig_U, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig_U.suptitle("Optimal Controls", fontsize=14)

    for i in range(2):
        if i == 0:  # Curvature
            axs[i].plot(s_traj[:-1], U[:, i], marker=marker, color='tab:orange', ms=marker_size)
            axs[i].set_ylabel(r"Input Value [$m^{-1}s^{-1}$]")
        if i == 1:  # Acceleration
            axs[i].plot(s_traj[:-1], U[:, i], marker=marker, color='tab:orange', ms=marker_size)
            axs[i].set_ylabel(r"Input Value [$m/s^2$]")
        axs[i].set_title(control_names[i])
        axs[i].set_xlabel(r'Distance [$m$]')
        axs[i].grid(True)
    plt.tight_layout()

    # --- Slack (S) ---
    plt.figure()
    plt.plot(s_traj[:-1], S * 3.6, marker=marker, color='tab:green', ms=marker_size)  # m/s to km/h
    plt.title("Optimal Velocity Slack Variable")
    plt.xlabel(r'Distance [$m$]')
    plt.ylabel(r"Value [$km/h$]")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


def recreate_optimal_trajectory(X, route):
    """
    Reconstructs and plots the path of the optimized trajectory (tracked from the GraphHopper reference path).
    It also plots the reference path and add visual road boundaries to verify the vehicle's tracking performance
    and path feasibility.
    """
    # Reference path
    from trajectory_planning import unpack_reference_path
    detailed_points, reference_path_spline, speed_limits, ref_s_values, s_total = unpack_reference_path(route)

    # Total trajectory time
    dt = 0.3
    total_time = (len(X) - 1) * dt

    # Interpolator s -> t (parametric)
    t_max = len(detailed_points) - 1
    t_values = np.linspace(0.0, t_max, len(detailed_points))
    s_to_t = interp1d(ref_s_values, t_values, kind='linear', fill_value='extrapolate')

    x_spl, y_spl = reference_path_spline

    # Transform solution (s, d) -> (x, y)
    traj_x = []
    traj_y = []

    # Road boundaries
    left_bound_x, left_bound_y = [], []
    right_bound_x, right_bound_y = [], []

    lane_width = 3.0  # meters

    # --- Re-create the reference path (spline) and optimal trajectory (solution) ---
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

        # Calculate vehicle's (optimal trajectory) position
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

    # Plot optimal trajectory
    velocities = X[:, 4] * 3.6  # m/s to km/h
    trajectory = plt.scatter(traj_x, traj_y, c=velocities, cmap='plasma', s=30, label='Optimal Trajectory')

    plt.colorbar(trajectory, label='Velocity (km/h)')
    plt.legend()
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Optimal Trajectory Visualization | Distance : {s_total:.1f} m | Time : {total_time:.1f} s')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def plot_tracking_results(hist_x, hist_t, hist_obs_s, hist_tl_state, trajectory, fsm, mpc):
    """
    Plots the tracking metrics from the tracking MPC during a full reference trajectory tracking simulation.
    """
    s_hist = hist_x[:, 0]
    trajectory_hist = np.array([trajectory.get_state(s) for s in s_hist])

    # --- Calculate Errors ---
    lat_err = hist_x[:, 1] - trajectory_hist[:, 1]
    ori_err = hist_x[:, 2] - trajectory_hist[:, 2]

    # --- Reference Trajectory Data ---

    # Time vector in seconds
    times = np.arange(len(hist_x)) * mpc.dt

    # Reference distance
    ds_grid = 0.5  # Finer grid for integration
    s_ref_vals = np.arange(0, trajectory.s_max, ds_grid)
    v_ref_vals = np.array([trajectory.get_state(s)[4] for s in s_ref_vals])
    v_ref_vals = np.maximum(v_ref_vals, 0.01)
    v_avg = 0.5 * (v_ref_vals[:-1] + v_ref_vals[1:])
    dt_vals = ds_grid / v_avg
    t_ref_vals = np.cumsum(dt_vals)
    t_ref_vals = np.insert(t_ref_vals, 0, 0.0)

    # Reference velocity
    vel_ref = trajectory_hist [:,4]

    # --- Identify Obstacle Events Indices ---

    # Dynamic Obstacle
    obs_s_arr = np.array(hist_obs_s, dtype=float)
    is_active = ~np.isnan(obs_s_arr)

    s_appear = None
    s_disappear = None
    t_obs_disappear = None

    if np.any(is_active):
        idx_appear = np.where(is_active)[0][0]
        idx_disappear = np.where(is_active)[0][-1]

        s_appear = hist_x[idx_appear, 0]
        s_disappear = hist_x[idx_disappear, 0]
        t_obs_disappear = times[idx_disappear]

    # Traffic Light
    tl_pos = fsm.tl_pos
    t_green = None
    # Find the first moment state is GREEN
    for i, state in enumerate(hist_tl_state):
        if state == 'GREEN':
            t_green = times[i]
            break

    # --- Trajectory Information ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # -- Distance --
    axs[0].plot(times, hist_x[:, 0], label='Actual (Tracking)', linewidth=2)
    axs[0].plot(t_ref_vals, s_ref_vals, 'r--', alpha=0.5, label='Reference (Optimal)')
    axs[0].set_ylabel('Distance [$m$]')
    axs[0].set_xlabel(rf'Step [$s$]')
    axs[0].set_title('Distance')
    # Add lines for Obstacle and Traffic Light
    if s_appear is not None:
        axs[0].axhline(y=s_appear, color='gray', linestyle=':', label='Obstacle Appears', linewidth=1.0)
        axs[0].axhline(y=s_disappear, color='gray', linestyle='--', label='Obstacle Disappears', linewidth=1.0)
    if fsm.traffic_light:
        axs[0].axhline(y=tl_pos, color='gray', linestyle='-.', label='Traffic Light', linewidth=1.0)
    axs[0].legend()
    axs[0].grid(True)

    # -- Velocity --
    axs[1].plot(s_hist, hist_x[:, 4] * 3.6, label='Actual (Tracking)', linewidth=2)
    axs[1].plot(s_hist, vel_ref * 3.6, 'r--', alpha=0.5, label='Reference (Optimal)')
    axs[1].set_ylabel('Velocity [$km/h$]')
    axs[1].set_xlabel('Distance [$m$]')
    axs[1].set_title('Velocity')
    # Add lines for Obstacle and Traffic Light
    if s_appear is not None:
        axs[1].axvline(x=s_appear, color='gray', linestyle=':', label='Obstacle Appears', linewidth=1.0)
        axs[1].axvline(x=s_disappear, color='gray', linestyle='--', label='Obstacle Disappears', linewidth=1.0)
    if fsm.traffic_light:
        axs[1].axvline(x=tl_pos, color='gray', linestyle='-.', label='Traffic Light', linewidth=1.0)
    axs[1].legend()
    axs[1].grid(True)

    # --- Deviation Errors ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # Lateral Error
    axs[0].plot(s_hist, lat_err)
    axs[0].set_ylabel('Deviation [$m$]')
    axs[0].set_xlabel('Distance [$m$]')
    axs[0].axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label="Ideal")
    axs[0].axhline(y=1.5, color='tomato', linestyle='--', label='Road Limits')
    axs[0].axhline(y=-1.5, color='tomato', linestyle='--')
    axs[0].set_title('Lateral Deviation Error')
    axs[0].legend()
    axs[0].grid(True)
    # Orientation Error
    axs[1].plot(s_hist, ori_err)
    axs[1].set_ylabel('Deviation [$rad$]')
    axs[1].set_xlabel('Distance [$m$]')
    axs[1].axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label="Ideal")
    axs[1].set_title('Orientation Deviation Error')
    axs[1].axis(ymin=-0.5, ymax=0.5)
    axs[1].legend()
    axs[1].grid(True)

    # --- CPU Time ---
    plt.figure(figsize=(10, 6))
    cpu_times_ms = hist_t * 1000
    plt.plot(cpu_times_ms)
    plt.axhline(150, color='tomato', linestyle='--', label='Limit')
    mean_cpu_ms = np.mean(cpu_times_ms)
    plt.axhline(float(mean_cpu_ms), color='gray', linestyle='--', label=f'Mean ({mean_cpu_ms:.1f} ms)')
    plt.ylabel('Time [$ms$]')
    plt.xlabel(rf'Step [$\Delta t={mpc.dt}s$]')
    plt.title('Solver Time')
    plt.ylim(0, 200)
    plt.legend()
    plt.grid(True)


    # --- Distance and Velocity wrt Dynamic Obstacle ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # -- Distance --
    min_len = min(len(hist_x), len(hist_obs_s))
    obs_dist = hist_obs_s[:min_len] - hist_x[:min_len, 0]
    obs_times = times[:min_len]
    obs_mask = ~np.isnan(obs_dist)
    if np.any(obs_mask):
        axs[0].plot(obs_times[obs_mask], obs_dist[obs_mask], linewidth=2, label='Relative Distance')
        axs[0].axhline(y=mpc.obstacle_safety_distance, color='tomato', linestyle='--',
                       label=f'Safety Limit ({mpc.obstacle_safety_distance}m)')
        if t_obs_disappear is not None:
            axs[0].axvline(x=t_obs_disappear, color='gray', linestyle='--', label='Obstacle Disappears')
            # Extend view 5 seconds past disappearance
            axs[0].set_xlim(left=obs_times[obs_mask][0], right=t_obs_disappear + 5.0)
        axs[0].set_title('Distance to Dynamic Obstacle')
        axs[0].set_xlabel(r'Time [$s$]')
        axs[0].set_ylabel('Distance [$m$]')
        axs[0].legend()
        axs[0].grid(True)
    else:
        axs[0].text(0.5, 0.5, "No Dynamic Obstacle Data", ha='center')

    # -- Velocity --
    if np.any(obs_mask):
        # Determine plotting indices
        start_idx = np.where(obs_mask)[0][0]
        end_idx = np.where(obs_mask)[0][-1]
        # Extend to include 10 seconds more data points (if available)
        extension_sec = 20.0
        extension_steps = int(extension_sec / mpc.dt)
        plot_end_idx = min(len(hist_x), end_idx + extension_steps)
        # Slice data
        plot_times = times[start_idx:plot_end_idx]
        plot_vel_ref = vel_ref[start_idx:plot_end_idx] * 3.6
        plot_vel_actual = hist_x[start_idx:plot_end_idx, 4] * 3.6
        # Reference velocity
        axs[1].plot(plot_times, plot_vel_ref, color='seagreen', linestyle=':', label='Reference Velocity')
        # Actual velocity
        axs[1].plot(plot_times, plot_vel_actual, linewidth=2, color='tab:orange', label='Actual Velocity')
        if t_obs_disappear is not None:
            axs[1].axvline(x=t_obs_disappear, color='gray', linestyle='--', label='Obstacle Disappears')
        axs[1].axhline(y=fsm.obs_v * 3.6, color='tomato', linestyle='--',
                       label=f'Obstacle Velocity ({fsm.obs_v * 3.6:.0f}km/h)')
        axs[1].set_title('Velocity Approaching Dynamic Obstacle')
        axs[1].set_xlabel(r'Time [$s$]')
        axs[1].set_ylabel('Velocity [$km/h$]')
        axs[1].legend()
        axs[1].grid(True)
    else:
        axs[1].text(0.5, 0.5, "No Dynamic Obstacle Data", ha='center')


    # --- Distance and Velocity wrt Traffic Light ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # -- Distance --
    tl_mask = (hist_x[:, 0] > (tl_pos - 200)) & (hist_x[:, 0] < (tl_pos + 20))
    tl_times = times[tl_mask]
    if fsm.traffic_light and np.any(tl_mask):
        dist_to_tl = tl_pos - hist_x[tl_mask, 0]
        axs[0].plot(tl_times, dist_to_tl, linewidth=2, label='Relative Distance')
        axs[0].axhline(y=mpc.obstacle_safety_distance, color='tomato', linestyle='--',
                       label=f'Safety Limit ({mpc.obstacle_safety_distance}m)')
        if t_green is not None:
            axs[0].axvline(x=t_green, color='gray', linestyle='--', label='Green Light')
        axs[0].set_title('Distance to Traffic Light')
        axs[0].set_xlabel(r'Time [$s$]')
        axs[0].set_ylabel('Distance [$m$]')
        axs[0].legend()
        axs[0].grid(True)
    else:
        axs[0].text(0.5, 0.5, "No Traffic Light Data", ha='center')

    # -- Velocity --
    if fsm.traffic_light and np.any(tl_mask):
        # Reference velocity
        vel_ref_tl = vel_ref[tl_mask] * 3.6
        axs[1].plot(tl_times, vel_ref_tl, color='seagreen', linestyle=':',
                    label='Reference Velocity')
        # Actual velocity
        vel_at_tl = hist_x[tl_mask, 4] * 3.6
        axs[1].plot(tl_times, vel_at_tl, linewidth=2, color='tab:orange', label='Actual Velocity')
        if t_green is not None:
            axs[1].axvline(x=t_green, color='gray', linestyle='--', label='Green Light')
        axs[1].set_title('Velocity Approaching Traffic Light')
        axs[1].set_xlabel(r'Time [$s$]')
        axs[1].set_ylabel('Velocity [$km/h$]')
        axs[1].legend()
        axs[1].grid(True)
    else:
        axs[1].text(0.5, 0.5, "No Traffic Light Data", ha='center')

    plt.tight_layout()
    plt.show()


def driving_animation(hist_x, hist_preds, hist_obs_s, hist_tl_state, trajectory, fsm):
    """
    Generates an animated visualization of the vehicle's trajectory tracking performance.

    The animation shows:
        - The vehicle's actual position and orientation.
        - The receding horizon prediction generated by the MPC at each step.
        - Dynamic obstacle movement (car), and static obstacle placement and state (traffic light), when present.
        - Real-time status text displaying distance traveled, current velocity, and distance to the dynamic obstacle.
        - The camera view dynamically follows the vehicle and remains zoomed on the driving environment.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('Global X [$m$]')
    ax.set_ylabel('Global Y [$m$]')

    # Road
    ax.plot(trajectory.global_x, trajectory.global_y, 'k--', alpha=0.2, label='Reference Path')

    # Actors
    ego_rect = Rectangle((0, 0), 4.0, 1.8, fc='blue', alpha=0.7, label='Vehicle')
    ax.add_patch(ego_rect)

    pred_line, = ax.plot([], [], 'c-o', linewidth=1, markersize=3, label='MPC Prediction')

    # Create Obstacle
    obs_rect = None
    if fsm.dynamic_obstacle:
        obs_rect = Rectangle((0, 0), 4.0, 1.8, fc='red', alpha=0.7, label='Obstacle')
        ax.add_patch(obs_rect)

    # Create Traffic Light
    tl_circle = None
    if fsm.traffic_light:
        # We don't add a label here to avoid the default square legend
        tl_circle = Circle((0, 0), 2.0, fc='gray')
        ax.add_patch(tl_circle)
        tl_glob = trajectory.get_global_pose(fsm.tl_pos, 4.0)
        tl_circle.center = (tl_glob[0], tl_glob[1])

    status_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=10,
                          bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # Precompute vehicle global positions
    ego_glob = []
    for i in range(len(hist_x)):
        p = trajectory.get_global_pose(hist_x[i, 0], hist_x[i, 1])
        p[2] += hist_x[i, 2]
        ego_glob.append(p)
    ego_glob = np.array(ego_glob)

    def update(frame):
        artists = [ego_rect, pred_line, status_text]

        # Base status text
        current_s = hist_x[frame, 0]
        velocity = hist_x[frame, 4] * 3.6
        text_str = f"Distance: {current_s:.0f}m\nVelocity: {velocity:.0f}km/h"

        # -- Update Vehicle --
        ex, ey, epsi = ego_glob[frame]
        cx, cy = -2.0, -0.9
        dx = cx * np.cos(epsi) - cy * np.sin(epsi)
        dy = cx * np.sin(epsi) + cy * np.cos(epsi)
        ego_rect.set_xy((ex + dx, ey + dy))
        ego_rect.angle = np.degrees(epsi)

        # -- Update Dynamic Obstacle --
        if obs_rect is not None:
            if frame < len(hist_obs_s):
                s_obs = hist_obs_s[frame]
                if not np.isnan(s_obs):
                    # Show obstacle
                    obs_rect.set_visible(True)
                    ox, oy, opsi = trajectory.get_global_pose(s_obs, 0.0)
                    dx = cx * np.cos(opsi) - cy * np.sin(opsi)
                    dy = cx * np.sin(opsi) + cy * np.cos(opsi)
                    obs_rect.set_xy((ox + dx, oy + dy))
                    obs_rect.angle = np.degrees(opsi)
                    # Update text with distance to obstacle
                    dist_to_obs = s_obs - current_s
                    text_str += f"\nDistance to Obstacle: {dist_to_obs:.1f}m"
                else:
                    obs_rect.set_visible(False)
            artists.append(obs_rect)

        # -- Update Traffic Light --
        if tl_circle is not None:
            if frame < len(hist_tl_state):
                state = hist_tl_state[frame]
                if state == 'RED':
                    tl_circle.set_facecolor('red')
                else:
                    tl_circle.set_facecolor('green')
            artists.append(tl_circle)

        # -- Update Prediction --
        if frame < len(hist_preds):
            px, py = [], []
            for j in range(len(hist_preds[frame])):
                ps, pd = hist_preds[frame][j, 0], hist_preds[frame][j, 1]
                g = trajectory.get_global_pose(ps, pd)
                px.append(g[0])
                py.append(g[1])
            pred_line.set_data(px, py)

        # Zoom
        zoom = 62
        ax.set_xlim(ex - zoom, ex + zoom)
        ax.set_ylim(ey - zoom, ey + zoom)

        # Set final text
        status_text.set_text(text_str)

        return artists

    ani = animation.FuncAnimation(fig, update, frames=range(0, len(hist_x), 2), interval=45, blit=False)

    # --- Legend ---
    handles, labels = ax.get_legend_handles_labels()

    if fsm.traffic_light:
        # Create a proxy artist: a Line2D with a circular marker, no line
        tl_proxy = Line2D([0], [0], marker='o', color='w', label='Traffic Light',
                          markerfacecolor='gray', markersize=10)
        handles.append(tl_proxy)
        labels.append('Traffic Light')
    plt.legend(handles=handles, labels=labels, loc='upper right')

    # --- Save the animation ---
    # try:
    #     ani.save('driving_simulation.gif', writer='pillow', fps=22)
    #     print("Animation saved successfully.")
    # except Exception as e:
    #     print(f"Could not save the animation. Ensure FFmpeg is installed. Error: {e}")

    plt.show()