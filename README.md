# dia_retrival

img2siftfeature_1.py
生成图像的sift特征

cluster_centers_2.py
sift特征层次k-means聚类, 得到视觉单词, i7-7700hq 8g内存, 约需要30min

sift2center_3.py
每个sift特征分配给一个视觉单词

retrival_4.py
计算每张图像和其他所有图像的距离

evaluate_5.py
根据上步的距离计算f-score和mAP
