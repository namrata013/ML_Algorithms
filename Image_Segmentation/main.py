import sys
import numpy as np
from pathlib import Path
import k_means
import img_seg

n_clusters = 3
iterations = 50
threshold = 0.0001
data = input('choose 1 for k-means on linearly separable data\n 2 for k-means on nonlinearly separable data\n 3 for k-means on real world data\n 4 for image segmentation\n')
while(int(data)>4 or int(data)<1):
    data = input("Try again : ")
d = int(data)

data_folder = " "
if d == 1:
    data_folder = Path("Data/linear_sep/")
    model = k_means.kmeans(data_folder,n_clusters,iterations,threshold)
    model.plot_rawdata()
    model.algorithm()
    model.plot_clusters()

if d==2:
    data_folder = Path("Data/nonlinear_sep/")
    model = k_means.kmeans(data_folder,n_clusters,iterations,threshold)
    model.plot_rawdata()
    model.algorithm()
    model.plot_clusters()

if d==3:
    data_folder = Path("Data/realworld_data/")
    model = k_means.kmeans(data_folder,n_clusters,iterations,threshold)
    model.plot_rawdata()
    model.algorithm()
    model.plot_clusters()

if d==4:
    image =  Path("Data/original1.jpg")
    model = img_seg.image_seg(image)
