import numpy as np
import pandas as pd
from utils import plot_outputs, plot_states, compare_UKFs, plot_params
from UnscentedKalmanFilter import UKFLithiumBattery, BatteryUKF 
from BatteryModels import BatteryCellPhy


def main(): 
    data_voltage = '../data/RW9_Voltage_Discharge_Reference/voltage_trace_01.csv'
    voltage_measurements = pd.read_csv(data_voltage)["voltage"].to_numpy()

    data_current = '../data/RW9_Current_Discharge_Reference/current_trace_01.csv'
    current_inputs = pd.read_csv(data_current)["current"].to_numpy()

    assert len(voltage_measurements) == len(current_inputs), "Voltage and current trace lengths do not match."

    # --- Initialize the battery model ---
    eod_threshold = 3.2     # [V]
    dt = 10.0               # [s]
    battery = BatteryCellPhy(dt=dt, eod_threshold=eod_threshold)
    x0 = battery.initialize()
    x = x0.copy()

    # --- Initialize UKF ---
    ukfNasa = UKFLithiumBattery(dt=dt, num_time_steps=len(current_inputs))
    # Optional: Set a custom initial state (if you want)
    # x0 = np.array([...])  # length = 8
    # ukf.reset(x0=x0)
    ukf = BatteryUKF(battery, dt=dt)

    # --- Run the battery model to generate voltage predictions ---
    x_model = np.zeros((len(current_inputs), len(x0)))
    V_model = np.zeros(len(current_inputs))               
    for k in range(len(current_inputs)):
        x_model[k] = x
        i_app = current_inputs[k]
        x = battery.getNextState(x, i_app)
        z = battery.getNextOutput(x, i_app)
        V_model[k] = z

    # --- Run the UKF over the input data ---
    xh_k = np.zeros((len(current_inputs), len(x0)))
    yh_k = np.zeros((len(current_inputs), 1))
    for u_k, y_k in zip(current_inputs, voltage_measurements):

        ukfNasa.step(u_k, y_k)

        x_est = ukf.step(y_k, u_k)
        xh_k[ukfNasa.step_counter-1] = x_est
        yh_k[ukfNasa.step_counter-1] = ukf.hx(x_est)[0]

    # --- Retrieve history ---
    # x_k = [T Vo Vsn Vsp qnB qnS qpB qpS]
    x_k, y_k = ukfNasa.get_history()
    t = np.arange(len(voltage_measurements)) * dt

    # --- Plot voltage prediction ---
    plot_outputs(t, voltage_measurements, V_model, y_k)
    # --- Plot states ---
    plot_states(t, x_model, x_k)
    # --- Plot Ro and qmax ---
    Vo = x_k[:, 1]
    Ro = Vo/current_inputs
    qmax = np.sum(x_k[:, 4:8], axis=1)
    plot_params(t, Ro, qmax)




if __name__ == "__main__":
    main()
