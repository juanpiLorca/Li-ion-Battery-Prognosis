import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from BatteryParameters import BatteryParameters
from BatteryModels import BatteryCellPhy


class UKFLithiumBattery:
    """
    Unscented Kalman Filter for a Lithium-Ion Battery Cell
    based on the model and UKF formulation of Daigle et al. (2012).
    """

    citation = {
        "authors": ["Matthew Daigle", "Abhinav Saxena", "Kai Goebel"],
        "title": "An Efficient Deterministic Approach to Model-based Prediction Uncertainty Estimation",
        "conference": "Annual Conference of the Prognostics and Health Management Society",
        "year": 2012,
        "organization": "NASA Ames Research Center",
        "url": "https://www.phmsociety.org/phm-conference/2012",
        "license": "Creative Commons Attribution 3.0"
    }

    def __init__(self, dt=1.0, eod_threshold=3.0, num_time_steps=1000):
        # Simulation and model interface
        self._dt = dt
        self._eod_threshold = eod_threshold
        self._model = BatteryCellPhy(dt=self._dt, eod_threshold=self._eod_threshold)
        x0 = self._model.initialize()

        # Model structure
        self._x_states = ['Tb', 'Vo', 'Vsn', 'Vsp', 'qnB', 'qnS', 'qpB', 'qpS']
        self._y_output = ['Vm']
        self._u_input = ['i_app']
        self.nx = len(self._x_states)
        self.ny = len(self._y_output)
        self.nu = len(self._u_input)

        # UKF scaling parameters (symmetric UT)
        self._kappa = 3.0 - self.nx
        self.w0 = self._kappa / (self.nx + self._kappa)
        self.wi = 1.0 / (2.0 * (self.nx + self._kappa))

        # State and output estimates
        self.x_apriori = np.copy(x0)
        self.x_aposteriori = np.copy(x0)
        self.y_apriori = np.zeros(self.ny, dtype=np.float32)
        self.y_aposteriori = np.zeros(self.ny, dtype=np.float32)
        self.innovation = np.zeros(self.ny, dtype=np.float32)

        # Covariance matrices & Kalman gain
        self.P0 = np.diag([1e-3] * self.nx)
        self.Pxx_apriori = np.copy(self.P0)
        self.Pxx_aposteriori = np.copy(self.P0)
        self.sqrtPxx_aposteriori = np.copy(self.Pxx_aposteriori)
        self.Pyy = np.zeros((self.ny, self.ny), dtype=np.float32)
        self.Pxy = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.K = np.zeros((self.nx, self.ny), dtype=np.float32)

        # Sigma point matrices
        self.x_sigma = np.zeros((self.nx, 2 * self.nx + 1), dtype=np.float32)
        self.y_sigma = np.zeros((self.ny, 2 * self.nx + 1), dtype=np.float32)

        # Process and measurement noise
        self.v = {
            'qpS': 1e-5, 'qpB': 1e-3, 'qnS': 1e-5, 'qnB': 1e-3,
            'Vo': 1e-10, 'Vsn': 1e-10, 'Vsp': 1e-10, 'Tb': 1e-6
        }
        self.n = {'Vm': 1e-3, 'Tbm': 1e-3}
        self.Q = np.diag([self.v[state] for state in self._x_states])
        self.R = np.array([[self.n['Vm']]])

        # Uncertainty descriptor for visualization/debug
        self._output_err = np.zeros((self.ny,), dtype=np.float32)
        self.y_std = np.zeros((self.ny,), dtype=np.float32)

        self.num_time_steps = num_time_steps
        self.step_counter = 0
        self._state_history = np.zeros((self.num_time_steps, self.nx), dtype=np.float32)
        self._output_history = np.zeros((self.num_time_steps, self.ny), dtype=np.float32)

    def reset(self, x0: np.ndarray = None):
        self.x_aposteriori = np.copy(x0)
        self.x_apriori = np.copy(x0)
        self.Pxx_apriori = np.copy(self.P0)
        self.Pxx_aposteriori = np.copy(self.P0)
        self.step_counter = 0
        self._state_history.fill(0)
        self._output_history.fill(0)

    def predictStep(self, u_k: float): 
        """
        Perform the UKF prediction step:
        x̂_k|k-1 = ∑ w_i * f(x_i, u)
        P_k|k-1 = Q + ∑ w_i * (x_i - x̂_k|k-1)(x_i - x̂_k|k-1)^T
        """
        # 1. Cholesky decomposition of P_aposteriori
        self.sqrtPxx_aposteriori = np.linalg.cholesky(self.Pxx_aposteriori)

        # 2. Generate sigma points
        self.x_sigma[:, 0] = self.x_aposteriori
        for i in range(self.nx):
            offset = self._kappa + self.nx
            sqrt_col = np.sqrt(offset) * self.sqrtPxx_aposteriori[:, i]
            self.x_sigma[:, i + 1] = self.x_aposteriori + sqrt_col
            self.x_sigma[:, i + 1 + self.nx] = self.x_aposteriori - sqrt_col

        # 3. Propagate through nonlinear model
        for i in range(2 * self.nx + 1):
            self.x_sigma[:, i] = self._model.getNextState(self.x_sigma[:, i], u_k)

        # 4. Predicted state mean
        self.x_apriori = self.w0 * self.x_sigma[:, 0] + \
            self.wi * np.sum(self.x_sigma[:, 1:], axis=1)

        # 5. Predicted covariance
        self.Pxx_apriori.fill(0.0)
        for i in range(2 * self.nx + 1):
            dX = self.x_sigma[:, i] - self.x_apriori
            w = self.w0 if i == 0 else self.wi
            self.Pxx_apriori += w * np.outer(dX, dX)
        self.Pxx_apriori += self.Q

        # Store predicted state
        self._state_history[self.step_counter % self.num_time_steps] = self.x_apriori

    def updateStep(self, u_k: float, y_k: float):
        """
        Perform the UKF measurement update step:
        ŷ_k|k-1 = ∑ w_i * h(x_i)
        Pyy = R + ∑ w_i * (y_i - ŷ)(y_i - ŷ)^T
        Pxy = ∑ w_i * (x_i - x̂)(y_i - ŷ)^T
        x̂_k|k = x̂_k|k-1 + K(y - ŷ)
        P_k|k = P_k|k-1 - K Pyy K^T
        """
        for i in range(2 * self.nx + 1):
            self.y_sigma[:, i] = self._model.getNextOutput(self.x_sigma[:, i], u_k)

        # 2. Predicted output mean
        self.y_apriori = self.w0 * self.y_sigma[:, 0] + \
            self.wi * np.sum(self.y_sigma[:, 1:], axis=1)

        self.Pyy.fill(0.0)
        for i in range(2 * self.nx + 1):
            dY = self.y_sigma[:, i] - self.y_apriori
            w = self.w0 if i == 0 else self.wi
            self.Pyy += w * np.outer(dY, dY)
        self.Pyy += self.R

        self.Pxy.fill(0.0)
        for i in range(2 * self.nx + 1):
            dX = self.x_sigma[:, i] - self.x_apriori
            dY = self.y_sigma[:, i] - self.y_apriori
            w = self.w0 if i == 0 else self.wi
            self.Pxy += w * np.outer(dX, dY)
        self.K = np.dot(self.Pxy, np.linalg.inv(self.Pyy))

        self._output_err = y_k - self.y_apriori
        self.x_aposteriori = self.x_apriori + np.dot(self.K, self._output_err)
        self.Pxx_aposteriori = self.Pxx_apriori - self.K @ self.Pyy @ self.K.T
        self.y_aposteriori = self._model.getNextOutput(self.x_aposteriori, u_k)

        self._output_history[self.step_counter % self.num_time_steps] = self.y_aposteriori
        # Step complete
        self.step_counter += 1

    def step(self, u_k, y_k): 
        """
        Predict and update step wrapper of the UKF
        """
        self.predictStep(u_k)
        self.updateStep(u_k, y_k)

    def get_history(self):
        n = min(self.step_counter, self.num_time_steps)
        if self.step_counter < self.num_time_steps:
            return self._state_history[:n], self._output_history[:n]
        else:
            start = self.step_counter % self.num_time_steps
            state_hist = np.vstack((self._state_history[start:], self._state_history[:start]))
            output_hist = np.vstack((self._output_history[start:], self._output_history[:start]))
            return state_hist, output_hist


class BatteryUKF:
    def __init__(self, model, dt=1.0):

        self.model = model
        self.dt = dt

        self.dim_x = 8
        self.dim_z = 1

        # Order of states must match the model
        self.state_order = ['Tb', 'Vo', 'Vsn', 'Vsp', 'qnB', 'qnS', 'qpB', 'qpS']
        # Process noise variances
        v = {
            'qpS': 1e-5,
            'qpB': 1e-3,
            'qnS': 1e-5,
            'qnB': 1e-3,
            'Vo': 1e-10,
            'Vsn': 1e-10,
            'Vsp': 1e-10,
            'Tb': 1e-6
        }
        # Sensor noise variance for voltage only
        n = {'Vm': 1e-3, 'Tbm': 1e-3}
        # Sigma points
        self.points = MerweScaledSigmaPoints(n=self.dim_x, alpha=0.1, beta=2.0, kappa=0.0)
        # UKF instance
        self.ukf = UKF(
            dim_x=self.dim_x,
            dim_z=self.dim_z,
            dt=self.dt,
            hx=self.hx,
            fx=self.fx,
            points=self.points
        )
        # Initial state
        self.ukf.x = self.model.initialize()
        # Initial estimate error covariance
        self.ukf.P *= 0.01
        # Process noise matrix Q
        self.ukf.Q = np.diag([
            v['Tb'],
            v['Vo'],
            v['Vsn'],
            v['Vsp'],
            v['qnB'],
            v['qnS'],
            v['qpB'],
            v['qpS']
        ])
        # Measurement noise matrix R (only voltage)
        self.ukf.R = np.array([[n['Vm']]])

    def fx(self, x, dt):
        """State transition"""
        return self.model.getNextState(x, self.current_input)

    def hx(self, x):
        """Measurement function"""
        return np.array([self.model.getNextOutput(x, self.current_input)])

    def step(self, z, u):
        """
        One UKF iteration step
        :param z: Measured voltage
        :param u: Applied current
        :return: Updated state estimate
        """
        self.current_input = u
        self.ukf.predict()
        self.ukf.update(np.array([z]))
        return self.ukf.x.copy()


if __name__ == "__main__":
    data_voltage = '../data/RW9_Voltage_Discharge_Reference/voltage_trace_01.csv'
    voltage_measurements = pd.read_csv(data_voltage)
    voltage_measurements = voltage_measurements["voltage"]
    data_current = '../data/RW9_Current_Discharge_Reference/current_trace_01.csv'
    current_inputs = pd.read_csv(data_current)
    current_inputs = current_inputs["current"]

    battey_parameters = BatteryParameters()
    dt = battey_parameters.sampleTime
    vEOD = battey_parameters.VEOD

    battery_model = BatteryCellPhy(dt=dt, eod_threshold=vEOD)
    ukf = BatteryUKF(battery_model, dt=dt)

    state_estimates = []
    output_estimates = []

    for z, u in zip(voltage_measurements, current_inputs):
        x_est = ukf.step(z, u)
        state_estimates.append(x_est)
        output_estimates.append(ukf.hx(x_est)[0])

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(current_inputs))*dt, voltage_measurements, label=r'$V(t)$', color='grey')
    plt.plot(np.arange(len(current_inputs))*dt, output_estimates, label=r'$\hat{V}(t)$', color='black', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title(f'Battery Voltage Estimation using UKF @ 1.0 (A)')
    plt.grid()
    plt.legend()
    plt.savefig('imgs/ukf_battery_voltage_estimation.pdf')
    plt.close()

    