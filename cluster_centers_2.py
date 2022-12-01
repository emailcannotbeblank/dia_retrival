import numpy as np
from sklearn.cluster import KMeans
import os

np.random.seed(666)

def get_sample_feature(all_sifts = None , sample_number = None):
    if type(all_sifts) == type(None):
        all_sifts = []
        sift_paths = os.listdir('./sifts/')
        for sift_path in sift_paths:
            if not '.jpg' in sift_path:
                continue
            sift = np.load('./sifts/' + sift_path)
            all_sifts.append(sift)
        all_sifts = np.vstack(all_sifts)

    if sample_number == None:
        # sample_number = all_sifts.shape[0]
        sample_sifts = all_sifts
    else:
        sample = np.random.rand((all_sifts.shape[0]))
        sample = np.argsort(sample)
        sample = sample[:sample_number]
        sample_sifts = all_sifts[sample]
    return sample_sifts

# following 4 function: my own kmeans
# in fact, useless
def cal_distance(data , clust_center):
    data = data.reshape((data.shape[0] , 1 , data.shape[-1]))
    clust_center = clust_center.reshape((1,clust_center.shape[0] , clust_center.shape[-1]))
    distance = (data - clust_center)**2
    distance = np.sum(distance , axis = -1)
    return distance


def cal_new_center(data , distance):
    # N * 1
    data_distibute = np.argmin(distance , axis = 1)
    # N * C
    data_distibute = np.eye(distance.shape[-1])[data_distibute]
    each_center_number = data_distibute.sum(axis = 0) + 1e-6
    data = data.reshape((data.shape[0] , 1 , data.shape[-1]))
    data_distibute = data_distibute.reshape((data_distibute.shape[0] ,data_distibute.shape[-1],1))
    new_center = data * data_distibute
    new_center = np.sum(new_center , axis = 0) / each_center_number.reshape((-1,1))
    return new_center
    

def cal_center(data , clust_center):
    iter_ = 0
    while True:
        iter_ += 1
        distance = cal_distance(data , clust_center)
        new_center = cal_new_center(data , distance)
        stop_condition = np.sum(np.abs(new_center - clust_center))
        clust_center = new_center
        if stop_condition < 1e-5 or iter_ > 100:
            print(iter_)
            break

    return clust_center


def cal_manysift_distance(data , center):
    distances = np.zeros((data.shape[0] , center.shape[0]))
    iter_number = 80000
    idx1 = 0
    while True:
        if idx1 + iter_number > data.shape[0]:
            idx2 = data.shape[0]
            distance = cal_distance(data[idx1:idx2 , :] , center)
            distances[idx1:idx2] = distance
            break 
        else:
            idx2 = idx1 + iter_number
            distance = cal_distance(data[idx1:idx2 , :] , center)
            distances[idx1:idx2] = distance
            idx1 += iter_number
    return distances


def test_plot():
    import matplotlib.pyplot as plt

    codebook_size = 10
    dim = 2
    feature_number = 10**2

    clust_center = np.random.rand(codebook_size , dim)
    data = np.random.rand(feature_number , dim)

    clust_center = cal_center(data , clust_center)
    distance = cal_distance(data , clust_center)
    data_distribute = np.argmin(distance , axis = 1)


    d0 = data[data_distribute == 0]
    d1 = data[data_distribute == 1]
    plt.scatter(d0[:,0] , d0[:,1] , c = 'red')
    plt.scatter(d1[:,0] , d1[:,1] , c = 'blue')
    # plt.scatter(clust_center[:,0] , clust_center[: , 1])
    plt.show()


def sklearn_kmean(data , kmeaner = None , codebook_size = None):
    if kmeaner == None:
        kmeaner = KMeans(n_clusters=codebook_size)
    kmeaner.fit(data)
    clust_center = kmeaner.cluster_centers_
    return clust_center



class Traverse:
    def __init__(self , depth , width , dim = 128):
        self.depth = depth 
        self.width = width
        self.dim = dim
        number_total_center = 0
        for idx in range(depth):
            number_total_center += width ** (idx + 1)
        self.number_total_center = number_total_center
        self.leaf_start = number_total_center - width ** depth + 1
        self.visited = np.zeros((number_total_center))
        self.visited[self.leaf_start:] = 1

        self.task_init()


    def k_is_small_son(self , k , width):
        small_son = (k % width) == 0
        return small_son

    def task_init(self):
        self.all_sift = get_sample_feature()
        self.all_sift = get_sample_feature(sample_number=self.all_sift.shape[0])
        self.all_sift = (self.all_sift - self.all_sift.mean()) / np.var(self.all_sift)**0.5
        self.all_centers = np.zeros((self.number_total_center , self.dim))
        self.select1 = [[] for _ in range(depth)]
        self.select2 = [[] for _ in range(depth)]
        self.d = 0

    def preprocess(self , k):
        if self.d == 0:
            data = self.all_sift
        else:
            idx1 = (k-1) % width
            idx1 = self.select1[self.d-1] == idx1 
            data = self.select2[self.d-1][idx1] 
        if data.shape[0] > 10**4:
            data_small = get_sample_feature(data , 10**4)
        else:
            data_small = data
        return data_small , data

    def doit(self , k):
        data , all_data = self.preprocess(k)
        if data.shape[0] <= self.width:
            clust_center = np.zeros((self.width , self.dim))
            clust_center[:data.shape[0] , :] = data
            data_distribute = np.arange(data.shape[0])
        else:
            kmeaner = KMeans(self.width)
            clust_center = sklearn_kmean(data , kmeaner=kmeaner , codebook_size=self.width)
            data_distribute = kmeaner.predict(all_data)
        idx = self.width*k 
        self.all_centers[idx:idx + width] = clust_center
        
        self.select1[self.d] = data_distribute
        self.select2[self.d] = all_data
        print(k)

    def traverse(self):
        k = 0
        while k < self.number_total_center:
            old_son = self.width * k + 1
            # small_son = self.width * (k + 1)
            parent = (k - 1) // self.width
            if self.visited[k] == 1:
                is_small_son = self.k_is_small_son(k , self.width)
                # return its parent or his little brother
                if is_small_son:
                    k = parent
                    self.d -= 1
                else:
                    k = k + 1
                continue 

            elif self.visited[k] == 0:
                self.visited[k] = 1
                self.doit(k)
                k = old_son
                self.d += 1




if __name__ == '__main__':
    depth = 5
    width = 10
    dim = 128

    tv = Traverse(depth,width,dim)
    tv.traverse()
    all_centers = tv.all_centers
    # data = tv.all_sift
    np.save('sifts/centers.npy' , all_centers)

