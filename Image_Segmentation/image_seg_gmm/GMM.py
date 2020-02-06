import KMeans
import statistics_func as sf 
import math
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
def get_Cov(Class_train,mean1,mean2,index1,index2):
	var=0
	for i in range(len(Class_train)):
		var=var+(Class_train[i][index1]-mean1)*(Class_train[i][index2]-mean2)
	var=var/len(Class_train)
	return var
def getCovMatrix(DATA,mean,diagonal=False):
	dim=len(DATA[0])
	matrix = [[0.0 for i in range(dim)] for j in range(dim)]
	if diagonal==False:
		for i in range(dim):
			for j in range(dim):
				matrix[i][j]=get_Cov(DATA,mean[i],mean[j],i,j)
	else:
		for i in range(dim):
			matrix[i][i]=get_Cov(DATA,mean[i],mean[j],i,i)
	if np.linalg.det(matrix)==0:
		# print "ASD"
		# print matrix
		for i in range(len(DATA[0])):
			for j in range(len(DATA[0])):
				if(matrix[i][j]==0 and i==j):
					matrix[i][j]=1.0
	# print np.linalg.det(matrix),"ASDdddddddddddddddddddddddddd"
	return np.array(matrix)
def prod(A,Sigma,B):
	Sigma=np.array(Sigma)
	temp=Sigma.dot(np.array(A))
	ret=0.0
	for i in range(len(B)):
		ret=ret+(B[i]*temp[i])
	return ret
def Normal_fn(x,mean,Sigma):
	y=[]
	for i in range(len(x)):
		y.append(x[i]-mean[i])
	return (1/(math.pow(2*3.14159265,(len(x)*1.0)/2.0))*(1.0/np.linalg.det(Sigma))*(math.pow(2.71828,-0.5*(prod(y,Sigma,y)))))
def GMMCluster(DATA,Num_of_Clusters,diagonal=False, kmeans=[]):#data for class not the entire dataset
	#Initialise
	if(kmeans==[]):
		means,Clusters=KMeans.KMeans(DATA,Num_of_Clusters)
	else:
		means, Clusters = kmeans[0],kmeans[1]
	Sigma=[]
	print ("KMeans Done")
	for i in range(Num_of_Clusters):
		Sigma.append(getCovMatrix(Clusters[i],means[i],diagonal))
	pi=[]
	for i in range(Num_of_Clusters):
		pi.append((1.0*len(Clusters[i]))/len(DATA))
	print ("INIT DONE")
	#--------------------------------------------------------
	Distortion=[]
	thresh=0.1
	while(len(Distortion)<5 or abs(Distortion[len(Distortion)-1]-Distortion[len(Distortion)-2])>thresh):
	# for i in range(1):		# Finding Gamma(n,k)
		Gamma=[]
		Clusters=[]
		for j in range(Num_of_Clusters):
			Clusters.append([])
		for i in range(len(DATA)):
			vec=[]
			for j in range(Num_of_Clusters):
				vec.append(0.0)
			SUM=0.000000000
			for j in range(Num_of_Clusters):
				# print np.linalg.det(Sigma[j]),"asdddddddd"
				# prod=1
				# for k in range(len(Sigma[j])):
				# 	prod*=Sigma[j][k][k]
					# print Sigma[j][k][k]
				# print np.linalg.det(Sigma[j])
				SUM=SUM+(pi[j]*multivariate_normal.pdf(DATA[i],mean=means[j],cov=Sigma[j],allow_singular=True))
			for j in range(Num_of_Clusters):
				vec[j]=(pi[j]*multivariate_normal.pdf(DATA[i],mean=means[j],cov=Sigma[j],allow_singular=True))/SUM
			Gamma.append(vec)
			# print vec,max(vec),vec.index(max(vec))
			Clusters[vec.index(max(vec))].append(DATA[i])
		#--------------------------------------------------------
		num=0.0
		for i in range(len(DATA)):
			for k in range(Num_of_Clusters):
				num=num+pi[k]*multivariate_normal.pdf(DATA[i],mean=means[k],cov=Sigma[k],allow_singular=True)
		Distortion.append(num)
		print (num)
		#--------------------------------------------------------
		#re-estimate mean,Sigma and pi
		#       mean
		for i in range(Num_of_Clusters):
			temp_mean=[]
			count=0.00
			for j in range(len(DATA[0])):
				temp_mean.append(0.0)
			for j in range(len(DATA)):
				for k in range(len(DATA[0])):
					temp_mean[k]+=DATA[j][k]*Gamma[j][i]
				count+=Gamma[j][i]
			for j in range(len(temp_mean)):
				temp_mean[j]=temp_mean[j]/count
			means[i]=temp_mean
		# print means
		#       Sigma
		for i in range(Num_of_Clusters):
			count=0.0
			temp_sigma=[]
			for j in range(len(Sigma[0])):
				x=[]
				for k in range(len(Sigma[0])):
					x.append(0.0)
				temp_sigma.append(x)
			for j in range(len(DATA)):
				count+=Gamma[j][i]
				temp_sigma=np.add(temp_sigma,Gamma[j][i]*np.outer(np.array(DATA[j])-np.array(means[i]),np.array(DATA[j])-np.array(means[i]))) 		
			temp_sigma=temp_sigma/count	
			if(diagonal==True or np.linalg.det(temp_sigma)<(7.65502597261e-50)):
				# print temp_sigma
				for a in range(len(Sigma[0])):
					for b in range(len(Sigma[0])):
						if(a!=b):
							temp_sigma[a][b]=0.0
						if(a==b and temp_sigma[a][b]<=1.0e-10):
							temp_sigma[a][b]=5.0
							# print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
				# print "APPLE is GREAT",np.linalg.det(temp_sigma)
			Sigma[i]=temp_sigma
		# print Sigma
		#       Pi
		for i in range(Num_of_Clusters):
			count=0.0
			for j in range(len(DATA)):
				count+=Gamma[j][i]
			pi[i]=count/len(DATA)
		# print pi
		# sf.plot(Clusters[0],'bo',"",True,means[0],Sigma[0])
		# sf.plot(Clusters[1],'co',"",True,means[1],Sigma[1])
		# sf.plot(Clusters[2],'ro',"",True,means[2],Sigma[2])
		# sf.plot(Clusters[3],'ko',"",True,means[3],Sigma[3])
		# plt.show()
	return means,Sigma,pi,Clusters