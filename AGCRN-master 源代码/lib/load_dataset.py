import os
import numpy as np

def load_st_dataset(dataset):
    #output B, N, D
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if dataset == 'PEMSD4':
        data_path = os.path.join(base_dir, 'data', 'PEMS04', 'pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join(base_dir, 'data', 'PEMS08', 'pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
