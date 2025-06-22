import os
import numpy as np

def load_st_dataset(dataset):
    #output B, N, D
    if dataset.upper() in ['PEMSD8', 'PEMS08']:
        data_path = os.path.join('data/PEMS08/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #only the first dimension, traffic flow data
    elif dataset.upper() in ['PEMSD4', 'PEMS04']:
        data_path = os.path.join('data/PEMS04/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
