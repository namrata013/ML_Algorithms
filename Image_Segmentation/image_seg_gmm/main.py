from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt 
import statistics_func as sf
import GMMClassifier
import GMM
from scipy.stats import multivariate_normal
import KMeans
print ("Cells")
files = os.listdir("Data/Train/7by7/")
data = []
for i in files:
    image = np.load("Data/Train/7by7/" + i)
    if(data==[]):
        data = image
    else:
        data = np.append(data, image, axis = 0)
    # sf.plot(data.tolist(), "c,")
    # plt.show()
    # print len(data.tolist())
data = data.tolist()
print ("Data Loaded")
cluster_centers, clusters = KMeans.KMeans(data, 3)
colors = ["c,", "b,", "r,"]
print ("KMeans done")
aPlot = plt.subplot(111)
for i in range(3):
    sf.plot(clusters[i], colors[i])
for i in range(3):
    sf.plot([cluster_centers[i]], "c*")
plt.show()
# print len(data.tolist())
files = os.listdir("Data/Test/7by7/")
TEST = []
for i in files:
    image = np.load("Data/Test/7by7/" + i)
    TEST.append(image.tolist())
for i in range(len(TEST)):
    cluster1,cluster2,cluster3 = [],[],[]
    for j in range(len(TEST[i])):
        cluster = KMeans.getCluster(cluster_centers, TEST[i][j])
        if(cluster == 0):
            cluster1.append([j/505, j%505])
        if(cluster == 1):
            cluster2.append([j/505, j%505])
        if(cluster == 2):
            cluster3.append([j/505, j%505])
    print (len(cluster1),len(cluster2),len(cluster3))
    sf.plot(cluster1,"bo")
    sf.plot(cluster2,"ro")
    sf.plot(cluster3,"go")
    plt.show()
GMM_center, GMM_sigma,GMM_pi, GMM_clusters = GMM.GMMCluster(data, 3, True, [cluster_centers, clusters])
for i in range(3):
    sf.plot(GMM_clusters[i], colors[i])
for i in range(3):
    sf.plot([GMM_center[i]], "c*")
cluster1,cluster2,cluster3=[],[],[]
for i in range(len(TEST)):
    cluster1,cluster2,cluster3 = [],[],[]
    for j in range(len(TEST[i])):
        index=-1
        MAX=-10000000000000000.0
        for l in range(3):
            if(GMM_pi[l]*multivariate_normal.pdf(TEST[i][j],mean=GMM_center[l],cov=GMM_sigma[l],allow_singular=True)>MAX):
                index=l
                MAX=GMM_pi[l]*multivariate_normal.pdf(TEST[i][j],mean=GMM_center[l],cov=GMM_sigma[l],allow_singular=True)
        if index==0:
            cluster1.append([j/505, j%505])
        if index==1:
            cluster2.append([j/505, j%505])
        if index==2:
            cluster3.append([j/505, j%505])
    sf.plot(cluster1,"bo")
    sf.plot(cluster2,"ro")
    sf.plot(cluster3,"go")
    plt.show()
    
