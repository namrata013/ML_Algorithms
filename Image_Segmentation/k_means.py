import sys
import math
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

class kmeans():
    max_iterations = 0
    threshold=0
    n_clusters=0
    data = []
    dist = []
    clusters = []
    cluster_centres = []
    def __init__(self,data_folder,n,i,t):
        self.max_iterations = i
        self.threshold = t
        self.n_clusters = n
        f1 = data_folder / "Class1.txt"
        f2 = data_folder / "Class2.txt"
        f3 = data_folder / "Class3.txt"
        f=open(f1,"r")
        for line in f:
            a,b=line.split()
            self.data.append([float(a),float(b)])
        f.close()
        f=open(f2,"r")
        for line in f:
            a,b=line.split()
            self.data.append([float(a),float(b)])
        f.close()
        f=open(f3,"r")
        for line in f:
            a,b=line.split()
            self.data.append([float(a),float(b)])
        f.close()
    def plot_rawdata(self):
        fig, ax =  plt.subplots()
        A=[]
        B=[]	
        for i in self.data:
            A.append(i[0])
            B.append(i[1])
        ax.scatter(A,B,alpha = 0.5) 
        folder = Path("Results/")
        plt.savefig(folder / 'RealWorld_Data_Plot.png', dpi=300, bbox_inches='tight')
        plt.show()
    def distance(self,A,B):
        ans=0.0
        for i in range(len(A)):
            ans=ans+(A[i]-B[i])*(A[i]-B[i])
        return math.sqrt(ans)
    def algorithm(self):
        temp=[]
        for i in self.data:
            temp.append(tuple(i))
        temp=list(set(temp))
        self.cluster_centres=random.sample(temp, self.n_clusters)# initial random points
        dimensions=len(self.data[0])
        while(len(self.dist)<2 or abs(self.dist[len(self.dist)-1]-self.dist[len(self.dist)-2])>self.threshold ):
            if(len(self.dist)==self.max_iterations):
                break
            distortion=0.0
            for i in range(self.n_clusters):
                self.clusters.append([])
            for i in range(len(self.data)):  
                index=-1
                min_dist=100000000000000000.0
                for j in range(self.n_clusters):
                    val=self.distance(self.data[i],self.cluster_centres[j])
                    if(val<min_dist):
                        min_dist=val
                        index=j
                distortion=distortion+self.distance(self.data[i],self.cluster_centres[index])
                self.clusters[index].append(self.data[i])
            self.dist.append(distortion)
            for i in range(self.n_clusters):
                mean=[] 
                for j in range(dimensions):
                    mean.append(0.00)
                for j in range(len(self.clusters[i])):
                    mean=[mean[k]+self.clusters[i][j][k] for k in range(dimensions)]
                for j in range(dimensions):
                    mean[j]=mean[j]/len(self.clusters[i])
                self.cluster_centres[i]=mean
    def plot_clusters(self):
        fig, ax =  plt.subplots()
        label = ['cluster1', 'cluster 2', 'cluster 3']
        color = ['r', 'g', 'b']
        for j in range(self.n_clusters):
            A=[]
            B=[]	
            for i in self.clusters[j]:
                A.append(i[0])
                B.append(i[1])
            x = [self.cluster_centres[j][0]]
            y = [self.cluster_centres[j][1]]
            ax.scatter(A,B,label=label[j], alpha = 0.5, color = color[j])
            # ax.plot(x,y,color='k',label='centroid')
        plt.legend()
        folder = Path("Results/")
        plt.savefig(folder / 'clusters_after_kmeans_real_world_data.png', dpi=300, bbox_inches='tight')
        plt.show()
