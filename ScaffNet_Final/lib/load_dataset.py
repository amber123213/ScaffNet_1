import os
import numpy as np

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4' or dataset == 'PEMS04':
        data_path = os.path.join('data/PEMS04/pems04.npz')
        data = np.load(data_path)['data']
    elif dataset == 'PEMSD8' or dataset == 'PEMS08':
        data_path = os.path.join('data/PEMS08/pems08.npz')
        data = np.load(data_path)['data']
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
