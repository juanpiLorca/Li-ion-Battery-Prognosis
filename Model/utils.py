import numpy as np 
import matplotlib.pyplot as plt

# Removes NaN values from a 1D numpy array.
def remove_nan(arr): 
    """Removes NaN values from a 1D numpy array."""
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    return arr[~np.isnan(arr)]


def plot_states(t, x_model, x_k):
    """
    Plots the 7 internal battery states: 3 voltage-related and 4 charge-related,
    comparing model ground truth vs UKF estimated states.

    :param t: Time vector (1D array)
    :param x_model: True model state history, shape (N, 8)
    :param x_k: UKF estimated state history, shape (N, 8)
    """
    Vlabels = [r"$V_o(t)$", r"$V_{sn}(t)$", r"$V_{sp}(t)$"]
    xV = x_model[:, 1:4]
    xV_k = x_k[:, 1:4]

    qlabels = [r"$q_{nB}(t)$", r"$q_{nS}(t)$", r"$q_{pB}(t)$", r"$q_{pS}(t)$"]
    xq = x_model[:, 4:8]
    xq_k = x_k[:, 4:8]

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i in range(3):
        axs[i].plot(t, xV[:, i], label="Model", color='tab:green', linestyle='--')
        axs[i].plot(t, xV_k[:, i], label="UKF", color='tab:orange')
        axs[i].set_ylabel(Vlabels[i], fontsize=12)
        axs[i].grid(True, linestyle=':', alpha=0.6)
        axs[i].legend(fontsize=10, loc='best')
    axs[-1].set_xlabel("Time [s]", fontsize=12)
    fig.suptitle("Voltage States: Model vs UKF", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("imgs/ukf_voltage_states.pdf", dpi=300)
    plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axs = axs.flatten()
    for i in range(4):
        axs[i].plot(t, xq[:, i], label="Model", color='tab:blue', linestyle='--')
        axs[i].plot(t, xq_k[:, i], label="UKF", color='tab:red')
        axs[i].set_ylabel(qlabels[i], fontsize=12)
        axs[i].grid(True, linestyle=':', alpha=0.6)
        axs[i].legend(fontsize=10, loc='best')
    axs[-2].set_xlabel("Time [s]", fontsize=12)
    axs[-1].set_xlabel("Time [s]", fontsize=12)
    fig.suptitle("Charge States: Model vs UKF", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("imgs/ukf_charge_states.pdf", dpi=300)
    plt.close()


def plot_outputs(t, V_meas, V_model, y_k):
    """
    Plots the battery voltage measurements, UKF estimated voltage, model voltage,
    and the squared estimation errors over time.

    :param t: Time vector (1D array)
    :param V_meas: Measured voltage (1D array)
    :param V_model: Model voltage (1D array)
    :param y_k: UKF estimated voltage (2D array, shape (N, 1))
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # --- Voltage estimation subplot ---
    axs[0].plot(t, V_meas, label=r"Measured Voltage", color='tab:blue', linewidth=1.5, alpha=0.9)
    axs[0].plot(t, y_k[:, 0], label=r"UKF Estimated Voltage: $\hat{V}(t)$", color='tab:orange', linewidth=2.0)
    axs[0].plot(t, V_model, label=r"Battery Model Voltage: $V(t)$", color='tab:green', linewidth=2.0, linestyle='--')

    # EOD threshold marker
    eod_idx = np.where(V_model < 3.2)[0]
    if len(eod_idx) > 0:
        eod_time = np.squeeze(t[eod_idx[0]]).item()
        axs[0].axvline(eod_time, color='gray', linestyle=':', linewidth=1.25)
        axs[0].text(eod_time, 3.2, 'EOD Threshold', rotation=90,
                    verticalalignment='bottom', color='gray', fontsize=9)

    axs[0].set_ylabel("Voltage [V]", fontsize=12)
    axs[0].set_title("UKF Battery Voltage Estimation vs Ground Truth", fontsize=14)
    axs[0].legend(fontsize=10, loc='best')
    axs[0].grid(True, linestyle=':', alpha=0.6)

    # --- Squared error subplot ---
    ukf_error_sq = (y_k[:, 0] - V_meas) ** 2
    model_error_sq = (V_model - V_meas) ** 2

    axs[1].plot(t, ukf_error_sq, label=r"UKF Squared Error: $(\hat{V} - V_{\mathrm{meas}})^2$", 
                color='tab:red', linewidth=1.8)
    axs[1].plot(t, model_error_sq, label=r"Model Squared Error: $(V - V_{\mathrm{meas}})^2$", 
                color='tab:purple', linewidth=1.8, linestyle='--')

    axs[1].set_xlabel("Time [s]", fontsize=12)
    axs[1].set_ylabel("Squared Error [V$^2$]", fontsize=12)
    axs[1].legend(fontsize=10, loc='best')
    axs[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('imgs/ukf_battery_voltage_estimation.pdf', dpi=300)
    plt.close()



def compare_UKFs(t, ukf_y1, ukf_y2, label1, label2, V_ref, eod_threshold=3.2):
    """
    Compares two UKF voltage estimates against a reference voltage.
    Includes a zoomed-in subplot for the first 800 seconds.

    :param t: Time vector (1D array or shapeable to 1D)
    :param ukf_y1: First UKF voltage estimate (1D array)
    :param ukf_y2: Second UKF voltage estimate (1D array)
    :param label1: Label for the first UKF line
    :param label2: Label for the second UKF line
    :param V_ref: Reference voltage (e.g., measured or model voltage)
    :param eod_threshold: Voltage threshold to mark end-of-discharge
    """
    t = np.squeeze(t)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # --- Full range plot ---
    axs[0].plot(t, V_ref, label="Measured Voltage", color='tab:blue', linewidth=1.5, alpha=0.9)
    axs[0].plot(t, ukf_y1, label=label1, color='tab:orange', linewidth=2.0)
    axs[0].plot(t, ukf_y2, label=label2, color='tab:green', linewidth=2.0, linestyle='--')

    eod_idx = np.where(V_ref < eod_threshold)[0]
    if len(eod_idx) > 0:
        eod_time = np.squeeze(t[eod_idx[0]]).item()
        axs[0].axvline(eod_time, color='gray', linestyle=':', linewidth=1.25)
        axs[0].text(eod_time, eod_threshold, 'EOD Threshold', rotation=90,
                    verticalalignment='bottom', color='gray', fontsize=9)

    axs[0].set_title("UKF Voltage Estimate Comparison (Full Range)", fontsize=14)
    axs[0].set_ylabel("Voltage [V]", fontsize=12)
    axs[0].legend(fontsize=10, loc='best')
    axs[0].grid(True, linestyle=':', alpha=0.6)

    # --- Zoomed-in plot (first 800s) ---
    zoom_mask = t <= 800
    t_zoom = t[zoom_mask]
    V_ref_zoom = V_ref[zoom_mask]
    ukf_y1_zoom = ukf_y1[zoom_mask]
    ukf_y2_zoom = ukf_y2[zoom_mask]

    axs[1].plot(t_zoom, V_ref_zoom, label="MeasuredVoltage", color='tab:blue', linewidth=1.5, alpha=0.9)
    axs[1].plot(t_zoom, ukf_y1_zoom, label=label1, color='tab:orange', linewidth=2.0)
    axs[1].plot(t_zoom, ukf_y2_zoom, label=label2, color='tab:green', linewidth=2.0, linestyle='--')

    axs[1].set_title("Zoomed-In View: First 800 Seconds", fontsize=13)
    axs[1].set_xlabel("Time [s]", fontsize=12)
    axs[1].set_ylabel("Voltage [V]", fontsize=12)
    axs[1].legend(fontsize=10, loc='best')
    axs[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('imgs/ukf_voltage_comparison.pdf', dpi=300)
    plt.close()

