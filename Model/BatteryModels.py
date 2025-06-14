import numpy as np 
from BatteryParameters import BatteryParameters


class RedlichKisterExpansion(): 
    def __init__(self, electrode='positive', init_params=None): 

        self.parameters = BatteryParameters() if init_params is None else init_params

        self.electrode = electrode.lower()
        if electrode == 'positive': 
            self.A = np.array([
                self.parameters.Ap0, 
                self.parameters.Ap1, 
                self.parameters.Ap2, 
                self.parameters.Ap3, 
                self.parameters.Ap4, 
                self.parameters.Ap5, 
                self.parameters.Ap6, 
                self.parameters.Ap7, 
                self.parameters.Ap8, 
                self.parameters.Ap9, 
                self.parameters.Ap10, 
                self.parameters.Ap11, 
                self.parameters.Ap12
            ])
        
        elif electrode == 'negative':
            self.A = np.array([
                self.parameters.An0, 
                self.parameters.An1, 
                self.parameters.An2, 
                self.parameters.An3, 
                self.parameters.An4, 
                self.parameters.An5, 
                self.parameters.An6, 
                self.parameters.An7, 
                self.parameters.An8, 
                self.parameters.An9, 
                self.parameters.An10, 
                self.parameters.An11, 
                self.parameters.An12
            ])

        else: 
            raise ValueError("Electrode must be either 'positive' or 'negative'.")
        
    def getVINT(self, xi): 
        """
        Computes the interfacial voltage based on the Redlich-Kister expansion.
        :param xi: State variable (e.g., concentration or state of charge)
        :return: Interfacial voltage
        """
        if not (0 <= xi <= 1):
            raise ValueError("xi must be between 0 and 1.")
        
        N = len(self.A)
        VINT = 0.0
        ne = self.parameters.ne
        F = self.parameters.F

        for k in range(N): 
            term = (1 / (ne * F)) * (self.A[k] * ((2 * xi - 1)**(k + 1) - (2 * xi * k * (1 - xi)) / ((2 * xi - 1)**(1 - k))))
            VINT += term

        return VINT


class BatteryCellPhy(): 
    def __init__(self, dt=1.0, eod_threshold=3.0, init_params=None):
       
        self.dt = dt 
        self.eod_th = eod_threshold

        self.inputs = ['i_app']
        self.states = ['tb', 'Vo', 'Vsn', 'Vsp', 'qnB', 'qnS', 'qpB', 'qpS']
        self.outputs = ['T', 'v']

        self.parameters = BatteryParameters() if init_params is None else init_params

        self.VINTp = RedlichKisterExpansion('positive', self.parameters)
        self.VINTn = RedlichKisterExpansion('negative', self.parameters)

    def initialize(self):
        """
        Computes the initial state x0 of the bettery model
        """
        x0 = np.array([
            self.parameters.x0['Tb'],   # Initial temperature
            self.parameters.x0['Vo'],   # Initial voltage
            self.parameters.x0['Vsn'],  # Initial surface voltage negative electrode
            self.parameters.x0['Vsp'],  # Initial surface voltage positive electrode
            self.parameters.x0['qnB'],  # Initial charge in bulk negative electrode
            self.parameters.x0['qnS'],  # Initial charge at surface negative electrode
            self.parameters.x0['qpB'],  # Initial charge in bulk positive electrode
            self.parameters.x0['qpS']   # Initial charge at surface positive electrode
        ])
        return x0

    def getNextState(self, x, u): 
        """
        Computes the next state of the battery model given the current state and input
        :param x: Current state vector
        :param u: Input vector (current applied)
        :return: Next state vector
        """
        # Unpack the current state
        Tb, Vo, Vsn, Vsp, qnB, qnS, qpB, qpS = x
        # Unpack the input
        i_app = u

        Tbdot = 0.0
        xpS = np.clip(qpS / self.parameters.qpSMax, 1e-8, 1 - 1e-6)
        xnS = np.clip(qnS / self.parameters.qnSMax, 1e-8, 1 - 1e-6)                 

        CnBulk = qnB/self.parameters.VolB
        CnSurface = qnS/self.parameters.VolS
        CpBulk = qpB/self.parameters.VolB
        CpSurface = qpS/self.parameters.VolS
        qdotDiffusionBSn = (CnBulk - CnSurface) / self.parameters.tDiffusion
        qdotDiffusionBSp = (CpBulk - CpSurface) / self.parameters.tDiffusion

        qnBdot = -qdotDiffusionBSn
        qpBdot = -qdotDiffusionBSp
        qnSdot = qdotDiffusionBSn - i_app
        qpSdot = qdotDiffusionBSp + i_app

        Jn0 = self.parameters.kn*(1-xnS)**self.parameters.alpha * (xnS)**(1 - self.parameters.alpha)
        Jp0 = self.parameters.kp*(1-xpS)**self.parameters.alpha * (xpS)**(1 - self.parameters.alpha)
        Jn = i_app/self.parameters.Sn
        Jp = i_app/self.parameters.Sp

        VoNominal = i_app*self.parameters.Ro
        VsnNominal = (self.parameters.R*Tb/(self.parameters.F*self.parameters.alpha)) * np.arcsinh(Jn/(2*Jn0))
        VspNominal = (self.parameters.R*Tb/(self.parameters.F*self.parameters.alpha)) * np.arcsinh(Jp/(2*Jp0))
        Vodot = (VoNominal - Vo) / self.parameters.to 
        Vsndot = (VsnNominal - Vsn) / self.parameters.tsn 
        Vspdot = (VspNominal - Vsp) / self.parameters.tsp

        xNew = np.array([
            Tb + Tbdot * self.dt,
            Vo + Vodot * self.dt,
            Vsn + Vsndot * self.dt,
            Vsp + Vspdot * self.dt,
            qnB + qnBdot * self.dt,
            qnS + qnSdot * self.dt,
            qpB + qpBdot * self.dt,
            qpS + qpSdot * self.dt
        ])
        return xNew
    
    def getNextOutput(self, x, u): 
        """
        Computes the next output of the battery model given the input
        :param u: Input vector (current applied)
        :return: Output vector (temperature and voltage)
        """
        # Unpack the current state
        Tb, Vo, Vsn, Vsp, qnB, qnS, qpB, qpS = x
        # Unpack the input
        i_app = u

        # Compute the interfacial voltages 
        xpS = qpS / self.parameters.qpSMax
        xnS = qnS / self.parameters.qnSMax
        if xnS >= 1.0:                              # Needed for the case when the negative electrode is fully charged
            xnS -= 1e-6                 
        VINTp = self.VINTp.getVINT(xpS)
        VINTn = self.VINTn.getVINT(xnS)

        R = self.parameters.R
        ne = self.parameters.ne
        F = self.parameters.F
        Vep = self.parameters.U0p + (R*Tb/(ne*F))*np.log((1-xpS)/xpS) + VINTp
        Ven = self.parameters.U0n + (R*Tb/(ne*F))*np.log((1-xnS)/xnS) + VINTn

        V = Vep - Ven - Vo - Vsn - Vsp
        return V

        