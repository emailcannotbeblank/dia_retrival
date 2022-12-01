import numpy as np
import os
import time

def cal_distances(query , invert_table , key_number , p = 1):
    # import pdb ; pdb.set_trace()
    distances = np.ones((key_number)) * 2
    K = len(invert_table)
    for j in range(K):
        if query[j] > 0:
            for img_feature in invert_table[j]:
                idx , value = img_feature
                if p == 1:
                    add = np.abs(query[j] - value) - value - query[j]
                else:
                    add = np.abs(query[j] - value)**p - value**p - query[j]**p
                distances[idx] += add
    return distances


width = 10
depth = 5


distribute = np.load('sifts/distribute.npy')
distribute += 1
for idx in range(distribute.shape[1]-1):
    distribute[:,idx + 1] += (distribute[:,idx] * width)
idx_start = distribute.min(axis = 0)
distribute -= idx_start
distribute = distribute[: , depth - 1]
distribute = distribute.astype(np.int32)

sifts = os.listdir('sifts/')
idx = 0
end_positions = []
for sift in sifts:
    if '.jpg' in sift and '.npy' in sift:
        sif = np.load('sifts/' + sift)
        idx += sif.shape[0]
        end_positions.append(idx)
        
img_number = len(end_positions)

img_feature = np.zeros(shape=(img_number, width ** depth) , dtype=np.int16)
idx_start = 0
for idx_img in range(img_number):
    idx_end = end_positions[idx_img]
    sift = distribute[idx_start:idx_end]
    for idx in range(sift.shape[0]):
        idx_code = sift[idx]
        img_feature[idx_img , idx_code] += 1
    
    idx_start = idx_end

# l1 norm
img_feature = img_feature / img_feature.sum(axis = 1).reshape(-1,1)
# # l2 norm
# img_feature_norm = (img_feature**2).sum(axis = 1)**0.5
# img_feature = img_feature / img_feature_norm.reshape(-1,1)

invert_table = [[] for _ in range(img_feature.shape[1])]
for idx_code in range(len(invert_table)):
    imgs = img_feature[:,idx_code]
    for idx_img in range(imgs.shape[0]):
        if imgs[idx_img] > 0:
            invert_table[idx_code].append([idx_img, imgs[idx_img]])

time1 = time.time()
retrival_result = np.zeros((img_number , img_number))
for idx_img in range(img_number):
    retrival = cal_distances(img_feature[idx_img] , invert_table ,img_number )
    retrival_result[idx_img , :] = retrival
    print(idx_img)
time2 = time.time()
print(time2 - time1)
np.save('sifts/retrival.npy' , retrival_result)
# test = np.argsort(retrival_result,axis = 1)[:,:4]
# retrival_result = np.argsort(retrival_result,axis = 1)

# p1 = (retrival_result[:,0] // 4) == (retrival_result[:,1] // 4)
# p2 = (retrival_result[:,0] // 4) == (retrival_result[:,2] // 4)
# p3 = (retrival_result[:,0] // 4) == (retrival_result[:,3] // 4)
# print(p1.mean() , p2.mean() , p3.mean())