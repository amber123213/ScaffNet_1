import numpy as np
import pandas as pd
import pickle
import sys

if __name__ == '__main__':
    # 默认参数
    distance_file = 'data/PEMS08/distance.csv'
    num_nodes = 170
    output_file = 'data/PEMS08/adj_mx.pkl'

    # 读取distance.csv
    df = pd.read_csv(distance_file)
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for row in df.values:
        if len(row) != 3:
            continue
        i, j = int(row[0]), int(row[1])
        A[i, j] = 1
        A[j, i] = 1

    # AGCRN格式通常为 (sensor_ids, sensor_id_to_ind, [adj_mx])
    sensor_ids = [str(i) for i in range(num_nodes)]
    sensor_id_to_ind = {str(i): i for i in range(num_nodes)}
    adj_mx = [A]

    with open(output_file, 'wb') as f:
        pickle.dump((sensor_ids, sensor_id_to_ind, adj_mx), f)
    print(f'adj_mx.pkl saved to {output_file}') 