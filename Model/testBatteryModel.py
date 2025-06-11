import argparse
import numpy as np
import matplotlib.pyplot as plt
from BatteryModels import BatteryCellPhy


def main(): 

    parser = argparse.ArgumentParser(description="Test Battery Model")
    help_str = "Simulation type: 1 for constant load, 2 for pulsed load, 3 for cut-off voltage"
    parser.add_argument('--simulation', type=int, required=True, default=1, help=help_str)
    args = parser.parse_args()

    battery = BatteryCellPhy(dt=1.0, eod_threshold=3.0)
    Ts = battery.dt                     # Sampling time in seconds
    VEOD = battery.eod_th               # End of discharge voltage threshold in [V]

    if args.simulation == 1: 
        # --------------------------------- Constant Load Simulation --------------------------------- #
        x0 = battery.initialize()
        x = x0.copy()
        print("Initial State X: "+25*"-")
        print(f"Tb: {x0[0]}")
        print(f"Vo: {x0[1]}")
        print(f"Vsn: {x0[2]}")
        print(f"Vsp: {x0[3]}")
        print(f"qnB: {x0[4]}")
        print(f"qnS: {x0[5]}")
        print(f"qpB: {x0[6]}")
        print(f"qpS: {x0[7]}")
        print(42*"-")

        # Simulate a simple scenario -------------------------------------------------
        # Battery dies at 3685 time steps @ Ts = 1.0s >>> Last 3685 seconds at 2 [A] load
        time_steps = 3600               
        i_app = 2.0                                       # Current applied in [A]    
        print(f"Simulating battery discharge with {time_steps} time steps at {i_app} (A) load...")
        u = np.ones(shape=(time_steps,)) * i_app  
        V = np.zeros(shape=(time_steps,))                 # Output voltage [V] array
        x_states = np.zeros(shape=(time_steps, len(x0)))  # State history for debugging

        for t in range(time_steps):
            x = battery.getNextState(x, u[t])
            x_states[t, :] = x
            z = battery.getNextOutput(x, u[t])
            V[t] = z

        # Plot the output over time
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(V))*Ts, V, label=r'$V(t)$', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(f'New Battery Discharge Profile @ {i_app} (A) Load')
        plt.axhline(y=VEOD, color='black', linestyle='--', label='End of Discharge Voltage Threshold')
        plt.grid()
        plt.legend()
        plt.savefig('imgs/battery_discharge_profile.pdf')
        plt.close()

        # Plot the states: charge in both positive and negative electrodes (bulk and surface)
        qnB = x_states[:, 4]           # Charge in bulk negative electrode
        qnS = x_states[:, 5]           # Charge in surface negative electrode
        qpB = x_states[:, 6]           # Charge in bulk positive electrode
        qpS = x_states[:, 7]           # Charge in surface positive electrode
        q_max = qnB + qnS + qpB + qpS  # Total charge in the battery
        xn = (qnB + qnS) / q_max       # Normalized charge in negative electrode
        xp = (qpB + qpS) / q_max       # Normalized charge in positive electrode

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(qnB))*Ts, qnB, label=r'$q_{n,b}$')
        plt.plot(np.arange(len(qnS))*Ts, qnS, label=r'$q_{n,s}$')
        plt.plot(np.arange(len(qpB))*Ts, qpB, label=r'$q_{p,b}$')
        plt.plot(np.arange(len(qpS))*Ts, qpS, label=r'$q_{p,s}$')
        plt.xlabel('Time (s)')
        plt.ylabel('Charge (C)')
        plt.title('Charge in Electrode @ 2 (A) Load')
        plt.legend()
        plt.grid()
        plt.savefig('imgs/battery_charge_states.pdf')
        plt.close()

        # Plot the normalized charge in both electrodes
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axs[0].plot(np.arange(len(xn))*Ts, xn, label=r'$x_n$', color='blue')
        axs[0].plot(np.arange(len(xp))*Ts, xp, label=r'$x_p$', color="#F06400")
        axs[0].set_ylabel('Normalized Charge')
        axs[0].set_title('Mol-fraction of Charge in Electrodes @ 2 (A) Load')
        axs[0].legend()
        axs[0].grid()
        axs[1].plot(np.arange(len(q_max))*Ts, q_max, label=r'$q^{max}$', color='blue')
        axs[1].set_ylim(np.max(q_max) - 1e-8, np.max(q_max) + 1e-8)
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Total Charge (C)')
        axs[1].set_title('Total Charge in Battery @ 2 (A) Load')
        axs[1].legend()
        axs[1].grid()
        plt.tight_layout()
        plt.savefig('imgs/battery_charge_mol_fractions_q_max.pdf')
        plt.close()

    if args.simulation == 2:
        # --------------------------------- Pulse Load Simulation --------------------------------- #
        x0 = battery.initialize()
        x = x0.copy()
        print("Initial State X: "+25*"-")
        print(f"Tb: {x0[0]}")
        print(f"Vo: {x0[1]}")
        print(f"Vsn: {x0[2]}")
        print(f"Vsp: {x0[3]}")
        print(f"qnB: {x0[4]}")
        print(f"qnS: {x0[5]}")
        print(f"qpB: {x0[6]}")
        print(f"qpS: {x0[7]}")
        print(42*"-")

        # Simulate a simple scenario -------------------------------------------------
        # Battery dies at 3685 time steps @ Ts = 1.0s >>> Last 3685 seconds at 2 [A] load
        time_steps = 14000               
        i_app = 1.0                                     # Current applied in [A]    
        print(f"Simulating battery discharge with {time_steps} time steps at pulsed {i_app} (A) load...")

        u = np.zeros(shape=(time_steps,))  # Initialize input current array
        cycle = int(10 * 60)
        for t in range(time_steps):
            if t % cycle < cycle / 2:
                u[t] = i_app
            else:
                u[t] = 0.001

        V = np.zeros(shape=(time_steps,))               # Output voltage [V] array

        for t in range(time_steps):
            x = battery.getNextState(x, u[t])
            z = battery.getNextOutput(x, u[t])
            V[t] = z

        # Plot the output over time
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(V))*Ts, V, label=r'$V(t)$', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(f'New Battery Discharge Profile @ {i_app} (A) Pulsed Load')
        plt.axhline(y=VEOD, color='black', linestyle='--', label='End of Discharge Voltage Threshold')
        plt.grid()
        plt.legend()
        plt.savefig('imgs/battery_pulsed_discharge_profile.pdf')
        plt.close()

    if args.simulation == 3:
        # --------------------------------- Cut-off Voltage Simulation --------------------------------- #
        x0 = battery.initialize()
        x = x0.copy()
        print("Initial State X: "+25*"-")
        print(f"Tb: {x0[0]}")
        print(f"Vo: {x0[1]}")
        print(f"Vsn: {x0[2]}")
        print(f"Vsp: {x0[3]}")
        print(f"qnB: {x0[4]}")
        print(f"qnS: {x0[5]}")
        print(f"qpB: {x0[6]}")
        print(f"qpS: {x0[7]}")
        print(42*"-")

        # Simulate a simple scenario -------------------------------------------------
        # Discharge @ 2 [A] load until the voltage reaches the cut-off threshold, then cut the current for 2000 steps   
        i_app = 2.0         # Current applied in [A] 
        time_steps = 2000       
        print(f"Simulating battery discharge until V(t)={VEOD} (V) at {i_app} (A) load, then cut the load for 2000 steps...")

        V = []              # Output voltage [V] array  
        cut_i_app = False
        while not cut_i_app:
            x = battery.getNextState(x, i_app)
            z = battery.getNextOutput(x, i_app)
            V.append(z)

            if z <= VEOD:
                cut_i_app = True
                print(f"Cutting current at time step {len(V)} with V(t)={z:.2f} V")

        for t in range(time_steps):
            x = battery.getNextState(x, 0.0)
            z = battery.getNextOutput(x, 0.0)
            V.append(z)

        # Plot the output over time
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(V))*Ts, V, label=r'$V(t)$', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(f'New Battery Discharge Profile @ {i_app} (A) Load')
        plt.axhline(y=VEOD, color='black', linestyle='--', label='End of Discharge Voltage Threshold')
        plt.grid()
        plt.legend()
        plt.savefig('imgs/battery_discharge_unitl_cut_off.pdf')
        plt.close()

    if args.simulation == 4: 
        # --------------------------------- Compare with Reference Discharge --------------------------------- #
        ref = np.load('DischargeReference/data_RW.npy')

        battery = BatteryCellPhy(dt=10.0, eod_threshold=3.2)
        x0 = battery.initialize()

        i_app = 1.0         # Current applied in [A]   
        print(f"Simulating battery discharge until V(t)={VEOD} (V) at {i_app} (A) load...")

        x = x0.copy()
        V = []              # Output voltage [V] array  

        cut_i_app = False
        while not cut_i_app:
            x = battery.getNextState(x, i_app)
            z = battery.getNextOutput(x, i_app)
            V.append(z)

            if z <= battery.eod_th:
                cut_i_app = True
                print(f"Cutting current at time step {len(V)} with V(t)={z:.2f} V")

        # Plot the output over time
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(ref))*battery.dt, ref, label='Reference Discharge', color='blue', linestyle=':')
        plt.plot(np.arange(len(V))*battery.dt, V, label=r'$V(t)$', color='grey', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(f'New Battery Discharge Profile @ {i_app} (A) Load')
        plt.axhline(y=battery.eod_th, color='r', linestyle='--', label='End of Discharge Voltage Threshold')
        plt.grid()
        plt.legend()
        plt.savefig('imgs/battery_discharge_reference.pdf')
        plt.close()


if __name__ == "__main__":
    main()