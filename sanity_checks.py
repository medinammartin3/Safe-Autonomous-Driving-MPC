import numpy as np

def reference_trajectory_check(optimizer, X, U, S, s_total):
    """
    Verifies the physical feasibility of the optimal reference trajectory.
    """
    print("\n=== SANITY CHECKS ===")

    passed = True

    # Thresholds
    DISTANCE_TOL = 0.5  # meters
    VELOCITY_TOL = 0.1  # km/h
    CONTROLS_TOL = 0.1
    LATERAL_DEV_TOL = 1.5  # meters (Max lateral deviation allowed)
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
        print(f'Full stop at the end : False --> Final velocity = {v_final * 3.6} km/h')
        passed = False
    else:
        print(f'Full stop at the end : True')

    # --- Reverse Driving ---
    min_v = np.min(X[:, 4])
    if min_v < -VELOCITY_TOL:
        print(f'Always non-negative velocity : False --> Minimum velocity = {min_v * 3.6} km/h')
        passed = False
    else:
        print(f'Always non-negative velocity : True')

    # --- Control Bounds ---
    u1_min, u2_min = optimizer.u_min
    u1_max, u2_max = optimizer.u_max

    # U1 (Curvature)
    u1_check = not (np.min(U[:, 0]) < u1_min - CONTROLS_TOL and np.max(U[:, 0]) > u1_max + CONTROLS_TOL)
    print(f'Curvature rate limits respected : {u1_check}')
    if passed:
        passed = u1_check

    # U2 (Acceleration)
    u2_check = not (np.min(U[:, 1]) < u2_min - CONTROLS_TOL and np.max(U[:, 1]) > u2_max + CONTROLS_TOL)
    print(f'Acceleration limits respected : {u2_check}')
    if passed:
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



def trajectory_tracking_check(tracker, hist_x, hist_u, hist_t, hist_obs_s, hist_tl_state, fsm, s_total):
    """
    Verifies the safety, performance, and real-time constraints of the reference trajectory tracking simulation.
    """
    print("\n=== SANITY CHECKS ===")

    hist_x = np.array(hist_x)
    hist_u = np.array(hist_u)
    hist_t = np.array(hist_t)

    passed = True

    # Tolerances
    LATERAL_LIMIT = 1.5  # Road boundary meters
    CONTROLS_TOL = 0.1
    CPU_LIMIT = 150  # ms
    SAFETY_DIST = 1.0  # Absolute collision threshold (not just comfort safety)

    # --- Destination Reached ---
    s_final = hist_x[-1, 0]
    if s_final < s_total - 1.0:
        print(f'Destination reached : False --> Stopped at {s_final:.1f}/{s_total:.1f} m')
        passed = False
    else:
        print(f'Destination reached : True')

    # --- Lateral Deviation ---
    max_dev = np.max(np.abs(hist_x[:, 1]))
    if max_dev > LATERAL_LIMIT:
        print(f'Stayed on road : False --> Max Deviation = {max_dev:.2f} m')
        passed = False
    else:
        print(f'Stayed on road : True')

    # --- Controls ---
    u1_min, u2_min = tracker.u_min
    u1_max, u2_max = tracker.u_max

    u1_vals = hist_u[:, 0]
    u2_vals = hist_u[:, 1]

    # Check if controls exceeded limits
    if (np.min(u1_vals) < u1_min - CONTROLS_TOL) or (np.max(u1_vals) > u1_max + CONTROLS_TOL):
        print(f'Steering controls within limits : False')
        passed = False
    else:
        print(f'Steering controls within limits : True')

    if (np.min(u2_vals) < u2_min - CONTROLS_TOL) or (np.max(u2_vals) > u2_max + CONTROLS_TOL):
        print(f'Acceleration controls within limits : False')
        passed = False
    else:
        print(f'Acceleration controls within limits : True')

    # --- Real-time Feasibility (CPU Time) ---
    max_cpu = np.max(hist_t)
    if max_cpu*1000 > CPU_LIMIT:
        passed = False
        print(f'Real-time constraint respected : False --> Max CPU time {max_cpu * 1000:.1f}ms > {CPU_LIMIT}ms')
    else:
        print(f'Real-time constraint respected : True')

    # --- Dynamic Obstacle Collision ---
    if fsm.dynamic_obstacle:
        # Filter steps where obstacle existed
        obs_s = np.array(hist_obs_s)
        valid_mask = ~np.isnan(obs_s)

        if np.any(valid_mask):
            # Calculate distance only where obstacle exists
            min_len = min(len(hist_x), len(obs_s))
            veh_s = hist_x[:min_len, 0]
            obs_s_valid = obs_s[:min_len]

            # Distance vehicle to obstacle
            dists = obs_s_valid[valid_mask[:min_len]] - veh_s[valid_mask[:min_len]]
            min_dist = np.min(dists)

            if min_dist < SAFETY_DIST:
                print(f'Dynamic Obstacle Avoided : False --> Min Distance = {min_dist:.2f} m')
                passed = False
            else:
                print(f'Dynamic Obstacle Avoided : True')

    # --- Traffic Light Respected ---
    if fsm.traffic_light:
        tl_pos = fsm.tl_pos

        # Check if we passed the traffic light while it was RED
        passed_tl_indices = np.where(hist_x[:, 0] > tl_pos)[0]

        violation = False
        if len(passed_tl_indices) > 0:
            first_pass_idx = passed_tl_indices[0]
            # Check state at the moment of passing
            if hist_tl_state[first_pass_idx] == 'RED':
                violation = True

        if violation:
            print(f'Traffic Light Respected : False (Ran a RED light)')
            passed = False
        else:
            print(f'Traffic Light Respected : True')

    print(f'===> Checks passed : {passed}')
    return passed