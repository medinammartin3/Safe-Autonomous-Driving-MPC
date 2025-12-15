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