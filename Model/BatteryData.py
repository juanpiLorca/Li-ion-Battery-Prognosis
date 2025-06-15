import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

DATA_PATH = '../data/'

BATTERY_FILES = {
    1: '3. Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    2: '3. Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    3: '2. Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    4: '2. Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    5: '2. Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    6: '2. Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    7: '3. Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    8: '3. Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    9: '1. Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW{}.mat',
    10: '1. Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW{}.mat',
    11: '1. Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW{}.mat',
    12: '1. Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW{}.mat'
}

class BatteryDataFile():
    def __init__(self, mat_file_path):
        mat_contents = loadmat(mat_file_path)

        self.procedure = mat_contents['data'][0,0]['procedure'][0]
        self.description = mat_contents['data'][0,0]['description'][0]

        self.headers = [n[0] for n in mat_contents['data'][0,0]['step'].dtype.descr]

        self.data = mat_contents['data'][0,0]['step'][0,:]
        self.num_steps = len(self.data)

        self.operation_type = np.array([v[0] for v in self.data['type']])

    def getDischarge(self, varnames, min_size=0, discharge_type=None):
        seq_sizes = np.array([len(x[0,:]) for x in self.data[np.where(self.operation_type=='D')[0]][varnames[0]]])

        index = seq_sizes>min_size

        if discharge_type is not None:
            index = index & (self.data[np.where(self.operation_type=='D')[0]]['comment']==discharge_type)

        # Find the max length of all sequences
        max_length = max(len(x[0,:]) for x in self.data[np.where(self.operation_type=='D')[0]][varnames[0]][index])

        ret = np.array([
            np.pad(np.asarray(x[0,:]), (0, max_length - len(x[0,:])), 'constant', constant_values=np.nan)
            for x in self.data[np.where(self.operation_type=='D')[0]][varnames[0]][index]
        ])

        max_length = 0
        for i in np.arange(1, len(varnames)):
            # Find the maximum length of the arrays in the sequences
            current_max_length = max(
                len(x[0, :]) for x in self.data[np.where(self.operation_type == 'D')[0]][varnames[i]][index]
            )
            
            # Update the maximum length found
            max_length = max(max_length, current_max_length)

        for i in np.arange(1,len(varnames)):
            ret = np.vstack([
                ret,
                np.array([
                    np.pad(np.asarray(x[0,:]), (0, max_length - len(x[0,:])), 'constant', constant_values=np.nan)
                    for x in self.data[np.where(self.operation_type=='D')[0]][varnames[i]][index]
                ])
            ])

        return ret

def getDischargeMultipleBatteries(data_path=BATTERY_FILES, varnames=['voltage', 'current', 'relativeTime'], discharge_type='reference discharge'):
    data_dic = {}

    for RWi,path in data_path.items():
        print('Loading data for RW{}...'.format(RWi))
        batterty_data = BatteryDataFile(DATA_PATH + data_path[RWi].format(RWi))
        data_dic[RWi] = batterty_data.getDischarge(varnames, discharge_type=discharge_type)

    
    return data_dic


if __name__ == '__main__':
    
    start = time.time()
    print('Loading battery data...')
    data_RW = getDischargeMultipleBatteries()
    end = time.time()
    print('Data loaded in {:.2f} seconds'.format(end - start))

    idx_example = 9
    print('Shape of data for RW{}:'.format(idx_example), data_RW[idx_example].shape)

    # --- Random Walk 9 data ---
    # >>> shaped (240, 757)
    # >>> First 80 sequences is voltage, next 80 is current, last 80 is relative time

    print('Plotting discharge sequence for RW{}...'.format(idx_example))
    print('Shape of the first discharge sequence for RW{}:'.format(idx_example), data_RW[idx_example][0, :].shape)
    plt.figure(figsize=(10, 5))

    for k in range(80):
        plt.plot(data_RW[idx_example][k, :], "k", alpha=0.5)

    plt.xlabel('Time Steps')
    plt.ylabel('Voltage (V)')
    plt.title('Discharge Sequence Voltage Profile for RW{}'.format(idx_example))
    plt.grid()
    plt.savefig('imgs/discharge_sequence_RW{}.pdf'.format(idx_example))
    plt.close()




