import numpy as np
from sklearn.cluster import KMeans
import time 
import os

from cluster_centers_2 import cal_distance,get_sample_feature



def cal_code(sift , centers , depth , width):
    result = np.ones(shape=(depth)) * (-1)
    idx_start = 0
    for i in range(depth):
        center = centers[idx_start:idx_start + width]
        distance = cal_distance(sift.reshape(1,-1) , center)
        idx_min = np.argmin(distance , axis = 1)
        result[i] = idx_min[0]
        idx_start = (idx_start + idx_min[0] + 1) * width
    return result


def cal_img_code(sifts , centers , depth , width):
    results = np.ones(shape=(sifts.shape[0],depth)) * (-1)
    for _ in range(sifts.shape[0]):
        result = cal_code(sifts[_] , centers , depth , width)
        results[_,:] = result
    return results





if __name__ == '__main__':

    # depth = 5
    depth = 5
    width = 10
    dim = 128

    all_centers = np.load('sifts/centers.npy')
    

    all_sift = get_sample_feature()
    all_sift = (all_sift - all_sift.mean()) / all_sift.var()**0.5
    path_sifts = os.listdir('sifts/')
    idx = 0
    end_positions = []
    for sift in path_sifts:
        if '.jpg' in sift:
            sif = np.load('sifts/' + sift)
            idx += sif.shape[0]
            end_positions.append(idx)

    data_distributes = np.ones(shape=(all_sift.shape[0],depth)) * (-1)
    idx_start = 0

    time1 = time.time()
    for idx_end in end_positions:
        sift = all_sift[idx_start:idx_end]
        data_distribute = cal_img_code(sift , all_centers , depth , width)
        data_distributes[idx_start:idx_end] = data_distribute
        print("%4.2f" % (idx_start / all_sift.shape[0] * 100))
        idx_start = idx_end
    time2 = time.time()
    np.save('sifts/distribute.npy' , data_distributes)
    print(time2 - time1)

    


