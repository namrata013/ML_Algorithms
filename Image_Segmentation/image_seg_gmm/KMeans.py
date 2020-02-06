import random
import math
def dist(A,B):
	ans=0.0
	for i in range(len(A)):
		ans=ans+(A[i]-B[i])*(A[i]-B[i])
	return math.sqrt(ans)
def KMeans(DATA,K):
	temp=[]
	for i in DATA:
		temp.append(tuple(i))
	temp=list(set(temp))
	cluster_centres=random.sample(temp, K)#	initial random points
	dimentions=len(DATA[0])
	D=[]
	# thresh=0.001
	thresh=10
	# print "Hello"
	while(len(D)<2 or abs(D[len(D)-1]-D[len(D)-2])>thresh ):
	# while len(D)<2 or (D[len(D)-1]-D[len(D)-2])>0:
		if(len(D)==50):
			break
		distortion=0.0
		Clusters=[]
		for i in range(K):
			Clusters.append([])
		for i in range(len(DATA)):	
			index=-1
			min_dist=100000000000000000.0
			for j in range(K):
				val=dist(DATA[i],cluster_centres[j])
				if(val<min_dist):
					min_dist=val
					index=j
			distortion=distortion+dist(DATA[i],cluster_centres[index])
			Clusters[index].append(DATA[i])
		D.append(distortion)
		print (distortion)
		for i in range(K):
			mean=[]	
			for j in range(dimentions):
				mean.append(0.00)
			for j in range(len(Clusters[i])):
				mean=[mean[k]+Clusters[i][j][k] for k in range(dimentions)]
			for j in range(dimentions):
				mean[j]=mean[j]/len(Clusters[i])
			cluster_centres[i]=mean
	return cluster_centres,Clusters
# data,test=sf.get_data("Class2.txt")
# KMeans(data,32)
def getCluster(cluster_centres, x):
	cluster_dist = []
	for i in cluster_centres:
		cluster_dist.append(dist(x,i))
	return cluster_dist.index(min(cluster_dist))