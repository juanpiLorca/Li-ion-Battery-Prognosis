import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from BatteryParameters import BatteryParameters
from BatteryModels import BatteryCellPhy


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

    