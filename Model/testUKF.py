import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from UnscentedKalmanFilter import UKFLithiumBattery  
from BatteryModels import BatteryCellPhy


def main(): 

    data_voltage = '../data/RW9_Voltage_Discharge_Reference/voltage_trace_01.csv'
    voltage_measurements = pd.read_csv(data_voltage)["voltage"].to_numpy()

    data_current = '../data/RW9_Current_Discharge_Reference/current_trace_01.csv'
    current_inputs = pd.read_csv(data_current)["current"].to_numpy()

    assert len(voltage_measurements) == len(current_inputs), "Voltage and current trace lengths do not match."

    # --- Initialize UKF ---
    dt = 10.0               # [s]
    ukf = UKFLithiumBattery(dt=dt, num_time_steps=len(current_inputs))
    # Optional: Set a custom initial state (if you want)
    # x0 = np.array([...])  # length = 8
    # ukf.reset(x0=x0)

    eod_threshold = 3.2     # [V]
    battery = BatteryCellPhy(dt=dt, eod_threshold=eod_threshold)
    x0 = battery.initialize()
    x = x0.copy()
    V = np.zeros(len(current_inputs))               

    for k in range(len(current_inputs)):
        i_app = current_inputs[k]
        x = battery.getNextState(x, i_app)
        z = battery.getNextOutput(x, i_app)
        V[k] = z

    # --- Run the UKF over the input data ---
    for u_k, y_k in zip(current_inputs, voltage_measurements):
        ukf.step(u_k, y_k)

    # --- Retrieve history ---
    x_k, y_k = ukf.get_history()

     # --- Plot voltage prediction ---
    t = np.arange(len(voltage_measurements)) * dt

    plt.figure(figsize=(12, 6))
    plt.plot(t, voltage_measurements, label="Measured Voltage", 
             color='tab:blue', linewidth=1.5, alpha=0.9)
    plt.plot(t, y_k[:, 0], label="UKF Estimated Voltage", 
             color='tab:orange', linewidth=2.0, linestyle='-')
    plt.plot(t, V, label="Battery Model Voltage", 
             color='tab:green', linewidth=2.0, linestyle='--')

    eod_idx = np.where(V < 3.2)[0]
    if len(eod_idx) > 0:
        plt.axvline(t[eod_idx[0]], color='gray', linestyle=':', linewidth=1)
        plt.text(t[eod_idx[0]], 3.25, 'EOD Threshold', rotation=90,
                 verticalalignment='bottom', color='gray')

    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Voltage [V]", fontsize=12)
    plt.title("UKF Battery Voltage Estimation vs Ground Truth", fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('imgs/ukf_battery_voltage_estimation.pdf', dpi=300)


if __name__ == "__main__":
    main()
