import json
import numpy as np
from scipy.interpolate import interp1d

class TrajectoryLoader:
    """
    Loads the offline JSON optimal reference trajectory (Output of trajectory_planning.py).

    Transforms the discrete trajectory points into a continuous reference signal, providing a framework to the
    controller to query the optimal state and control inputs at any arbitrary distance s.
    """

    def __init__(self, json_file):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found : {json_file}.")

        # Load offline optimal trajectory
        # X = [s, d, o, k, v]
        self.X_ref = np.array(data['X'])
        # U = [u1 : curvature rate, u2 : acceleration]
        self.U_ref = np.array(data['U'])

        # Enforce strict monotonicity for s to allow interpolation
        s_values = self.X_ref[:, 0].copy()
        for i in range(1, len(s_values)):
            if s_values[i] <= s_values[i - 1]:
                s_values[i] = s_values[i - 1] + 1e-5

        # --- Reconstruct global geometry for animation ---
        # Integrate curvature k to get Global Psi, then integrate Psi to get X, Y.
        self.global_x = [0.0]
        self.global_y = [0.0]
        self.global_psi = [0.0]

        for i in range(1, len(s_values)):
            ds = s_values[i] - s_values[i - 1]
            k = self.X_ref[i - 1, 3]  # Curvature at start of step

            # Current heading
            psi_old = self.global_psi[-1]

            # Next heading
            psi_new = psi_old + k * ds

            # Use average heading for position update
            psi_avg = (psi_old + psi_new) / 2.0

            # Update position
            x_new = self.global_x[-1] + np.cos(psi_avg) * ds
            y_new = self.global_y[-1] + np.sin(psi_avg) * ds

            # Add new states
            self.global_psi.append(psi_new)
            self.global_x.append(x_new)
            self.global_y.append(y_new)

        self.global_x = np.array(self.global_x)
        self.global_y = np.array(self.global_y)
        self.global_psi = np.array(self.global_psi)

        # --- Interpolators ---

        # State
        self.interp_d = interp1d(s_values, self.X_ref[:, 1], kind='linear', fill_value="extrapolate")
        self.interp_o = interp1d(s_values, self.X_ref[:, 2], kind='linear', fill_value="extrapolate")
        self.interp_k = interp1d(s_values, self.X_ref[:, 3], kind='linear', fill_value="extrapolate")
        self.interp_v = interp1d(s_values, self.X_ref[:, 4], kind='linear', fill_value="extrapolate")

        # Controls
        u_len = len(self.U_ref)
        limit = min(len(s_values), u_len)
        s_u = s_values[:limit]
        self.interp_u1 = interp1d(s_u, self.U_ref[:limit, 0], kind='linear', fill_value="extrapolate")
        self.interp_u2 = interp1d(s_u, self.U_ref[:limit, 1], kind='linear', fill_value="extrapolate")

        # Global
        self.interp_Gx = interp1d(s_values, self.global_x, kind='linear', fill_value="extrapolate")
        self.interp_Gy = interp1d(s_values, self.global_y, kind='linear', fill_value="extrapolate")
        self.interp_Gpsi = interp1d(s_values, self.global_psi, kind='linear', fill_value="extrapolate")

        self.s_max = s_values[-1]

    def get_state(self, s):
        """
        Returns optimal state [s, d, o, k, v] at distance s.
        """
        if s >= self.s_max:
            return self.X_ref[-1]
        else:
            return np.array([s, float(self.interp_d(s)), float(self.interp_o(s)), float(self.interp_k(s)), float(self.interp_v(s))])

    def get_control(self, s):
        """
        Returns optimal controls [u1, u2] at distance s.
        """
        if s >= self.s_max:
            return np.array([0.0, 0.0])
        else:
            return np.array([float(self.interp_u1(s)), float(self.interp_u2(s))])

    def get_global_pose(self, s, d):
        """
        Map Frenet (s,d) to Global (X,Y,Psi) for tracking animation.
        """
        if s > self.s_max: s = self.s_max
        xr = float(self.interp_Gx(s))
        yr = float(self.interp_Gy(s))
        psi_r = float(self.interp_Gpsi(s))

        # Transformation
        x_glob = xr - d * np.sin(psi_r)
        y_glob = yr + d * np.cos(psi_r)
        return np.array([x_glob, y_glob, psi_r])