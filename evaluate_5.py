import numpy as np
import os
from shutil import copyfile

def score2truefalse(retrival_result):
    retrival_result = np.argsort(retrival_result,axis = 1)
    retrival_result = retrival_result // 4
    retrival_truefalse = np.zeros(shape=(retrival_result.shape[0] , retrival_result.shape[1] - 1) , dtype=np.bool8)
    for idx in range(1,retrival_result.shape[1]):
        retrival_truefalse[:,idx-1] = retrival_result[:,0] == retrival_result[:,idx]
    return retrival_truefalse

def precision(retrival_result , number):
    retrival_result = retrival_result[:,:number]
    return retrival_result.mean()

def recall(retrival_result , number , gt_number):
    retrival_result = retrival_result[:,:number]
    retrival_result = retrival_result.sum(axis = 1) / gt_number
    return retrival_result.mean()

def f_score(p , r):
    return (2*p*r) / (p + r + 1e-6)

def mAP(retrival_result , gt_number):
    maps = np.zeros(shape=(retrival_result.shape[0]))
    for i in range(retrival_result.shape[0]):
        number = 1
        idx = 0
        while number <= gt_number:
            if retrival_result[i , idx] == 1:
                maps[i] += (number / (idx + 1))
                number += 1
            idx += 1
    return maps.mean()/3


# # show some result
# retrival_result = np.load('sifts/retrival4.npy')
# retrival_result = np.argsort(retrival_result,axis = 1)

# img_path_list = os.listdir('Image')
# idx_random_choice = np.random.rand(len(img_path_list))
# # show 5 images
# idx_random_choice = np.argsort(idx_random_choice)[:5]
# retrival_result = retrival_result[idx_random_choice]
# for i in range(5):
#     result = retrival_result[i]
#     for j in range(4):
#         img_source = 'Image/' + img_path_list[result[j]]
#         img_target = 'sifts/' + str(4 * i + j) + '.jpg'
#         copyfile(img_source , img_target)



retrival_result = np.load('sifts/retrival.npy')

retrival_result = score2truefalse(retrival_result) 
p = precision(retrival_result,3)
r = recall(retrival_result , 3 , 3)
f = f_score(p , r)
m = mAP(retrival_result , 3)

print(f , m)

# retrival_result = np.argsort(retrival_result,axis = 1)
# p1 = (retrival_result[:,0] // 4) == (retrival_result[:,1] // 4)
# p2 = (retrival_result[:,0] // 4) == (retrival_result[:,2] // 4)
# p3 = (retrival_result[:,0] // 4) == (retrival_result[:,3] // 4)
# print(p1.mean() , p2.mean() , p3.mean())
