import pandas as pd
import matplotlib.pyplot as plt
from UnscentedKalmanFilter import UKFLithiumBattery  

def main(): 

    data_voltage = '../data/RW9_Voltage_Discharge_Reference/voltage_trace_01.csv'
    voltage_measurements = pd.read_csv(data_voltage)["voltage"].to_numpy()

    data_current = '../data/RW9_Current_Discharge_Reference/current_trace_01.csv'
    current_inputs = pd.read_csv(data_current)["current"].to_numpy()

    assert len(voltage_measurements) == len(current_inputs), "Voltage and current trace lengths do not match."

    # --- Initialize UKF ---
    dt = 10.0  # seconds
    ukf = UKFLithiumBattery(dt=dt, num_time_steps=len(current_inputs))

    # Optional: Set a custom initial state (if you want)
    # x0 = np.array([...])  # length = 8
    # ukf.reset(x0=x0)

    # --- Run the UKF over the input data ---
    for u_k, y_k in zip(current_inputs, voltage_measurements):
        ukf.step(u_k, y_k)

    # --- Retrieve history ---
    x_hist, y_hist = ukf.get_history()

    # --- Plot voltage prediction ---
    plt.figure(figsize=(10, 5))
    plt.plot(voltage_measurements, label="Measured Voltage", alpha=0.7)
    plt.plot(y_hist[:, 0], label="UKF Estimated Voltage", linestyle="--")
    plt.xlabel("Time Step")
    plt.ylabel("Voltage [V]")
    plt.title("UKF Battery Voltage Tracking")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
