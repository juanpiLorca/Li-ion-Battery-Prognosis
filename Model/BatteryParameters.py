class BatteryParameters:
    # Define class attributes for constants and parameters
    sampleTime = 1.0

    capacity = 2.2           # battery capacity in Ah
    qMobile = capacity*3600  # maximum mobile charge (Li ions) in the battery ==> Related to battery capacity by qMobile = C * 3600, where C is the capacity in Ah
    xnMax = 0.6              # maximum mole fraction (neg electrode)
    xnMin = 0                # minimum mole fraction (neg electrode)
    xpMax = 1.0              # maximum mole fraction (pos electrode)
    xpMin = 0.4              # minimum mole fraction (pos electrode) -> note xn+xp=1
    qMax = qMobile / (xnMax - xnMin)  # note qMax = qn + qp
    Ro = 0.117215            # for Ohmic drop (current collector resistances plus electrolyte resistance plus solid phase resistances at anode and cathode)

    # Constants of nature
    R = 8.3144621          # universal gas constant, J/K/mol
    F = 96487              # Faraday's constant, C/mol
    ne = 1                 # number of electrons transferred in the reaction

    # Li-ion parameters
    alpha = 0.5            # anodic/cathodic electrochemical transfer coefficient
    Sn = 0.000437545       # surface area (- electrode)
    Sp = 0.00030962        # surface area (+ electrode)
    kn = 2120.96           # lumped constant for BV (- electrode)
    kp = 248898            # lumped constant for BV (+ electrode)
    Vol = 2e-5             # total interior battery volume/2 (for computing concentrations)
    VolSFraction = 0.1     # fraction of total volume occupied by surface volume

    # Volumes (total volume is 2*P.Vol), assume volume at each electrode is the same and the surface/bulk split is the same for both electrodes
    VolS = VolSFraction * Vol  # surface volume
    VolB = Vol - VolS         # bulk volume

    # Set up charges (Li ions)
    qpMin = qMax * xpMin           # min charge at pos electrode
    qpMax = qMax * xpMax           # max charge at pos electrode
    qpSMin = qpMin * VolS / Vol    # min charge at surface, pos electrode
    qpBMin = qpMin * VolB / Vol    # min charge at bulk, pos electrode
    qpSMax = qpMax * VolS / Vol    # max charge at surface, pos electrode
    qpBMax = qpMax * VolB / Vol    # max charge at bulk, pos electrode
    qnMin = qMax * xnMin           # max charge at neg electrode
    qnMax = qMax * xnMax           # max charge at neg electrode
    qnSMax = qnMax * VolS / Vol    # max charge at surface, neg electrode
    qnBMax = qnMax * VolB / Vol    # max charge at bulk, neg electrode
    qnSMin = qnMin * VolS / Vol    # min charge at surface, neg electrode
    qnBMin = qnMin * VolB / Vol    # min charge at bulk, neg electrode
    qSMax = qMax * VolS / Vol      # max charge at surface (pos and neg)
    qBMax = qMax * VolB / Vol      # max charge at bulk (pos and neg)

    # Time constants
    tDiffusion = 7e6  # diffusion time constant (increasing this causes decrease in diffusion rate)
    to = 6.08671      # for Ohmic voltage
    tsn = 1001.38     # for surface overpotential (neg)
    tsp = 46.4311     # for surface overpotential (pos)

    # Redlich-Kister parameters (positive electrode)
    U0p = 4.03
    Ap0 = -31593.7
    Ap1 = 0.106747
    Ap2 = 24606.4
    Ap3 = -78561.9
    Ap4 = 13317.9
    Ap5 = 307387
    Ap6 = 84916.1
    Ap7 = -1.07469e+06
    Ap8 = 2285.04
    Ap9 = 990894
    Ap10 = 283920
    Ap11 = -161513
    Ap12 = -469218

    # Redlich-Kister parameters (negative electrode)
    U0n = 0.01
    An0 = 86.19
    An1 = 0
    An2 = 0
    An3 = 0
    An4 = 0
    An5 = 0
    An6 = 0
    An7 = 0
    An8 = 0
    An9 = 0
    An10 = 0
    An11 = 0
    An12 = 0

    # End of discharge voltage threshold
    VEOD = 3.0

    # Default initial conditions (fully charged)
    def __init__(self):
        # Initialize initial conditions
        self.x0 = {
            'qpS': self.qpSMin,
            'qpB': self.qpBMin,
            'qnS': self.qnSMax,
            'qnB': self.qnBMax,
            'Vo': 0,
            'Vsn': 0,
            'Vsp': 0,
            'Tb': 292.1  # in K, about 18.95 C
        }

        # Process noise variances
        self.v = {
            'qpS': 1e-5,
            'qpB': 1e-3,
            'qnS': 1e-5,
            'qnB': 1e-3,
            'Vo': 1e-10,
            'Vsn': 1e-10,
            'Vsp': 1e-10,
            'Tb': 1e-6
        }

        # Sensor noise variances
        self.n = {
            'Vm': 1e-3,
            'Tbm': 1e-3
        }


if __name__ == "__main__":
    # Example usage
    P = BatteryParameters()
    print("Initial conditions:", P.x0)
    print("Process noise variances:", P.v)
    print("Sensor noise variances:", P.n)
    print("Maximum charge at positive electrode surface:", P.qpSMax)
    print("Minimum charge at negative electrode bulk:", P.qnBMin)