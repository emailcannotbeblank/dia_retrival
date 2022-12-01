import numpy as np
import cv2
import os


def cal_sift_feature():
    sift = cv2.SIFT_create()
    # img = cv2.imread('Image/ukbench00003.jpg',0)
    # kp , des = sift.detectAndCompute(img , None)
    img_list = os.listdir('./Image/')
    for img_path in img_list:
        if img_path[-4:] == '.jpg':
            img = cv2.imread('./Image/' + img_path)
            kp , des = sift.detectAndCompute(img , None)
            np.save('./sifts/' + img_path + '.npy' , des)
            print('img %s get %d features' % (img_path , des.shape[0]))



if __name__ == '__main__':
    cal_sift_feature()

