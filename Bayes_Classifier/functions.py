import sys
import numpy as np
import math
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
# DATASET[i][j][k] 
# i class index
# j datapoint index
# k feature index

def getdata(file):
    f = open(file, "r")
    train =[]
    test = []
    X =[]
    for line in f:
        lst = line.split()
        X.append(lst)
    train=X[:int(len(X)*(0.75))]
    test=X[int(len(X)*0.75):]
    f.close()
    test = np.array(test)
    train = np.array(train)
    train = train.astype(np.float)
    test = test.astype(np.float)
    return train, test

def Mean(Class_train):
    A=np.zeros(len(Class_train[0]))
    for i in Class_train :
        for j in range(len(Class_train[0])):
            A[j]=A[j]+i[j]
    for i in range(len(A)):
        A[i]=A[i]/len(Class_train)
    return A

def get_Cov(Class_train,mean1,mean2,index1,index2):
    var=0
    for i in range(len(Class_train)):
        var=var+(Class_train[i][index1]-mean1)*(Class_train[i][index2]-mean2)
    var=var/len(Class_train)
    return var

def cov_mat(Class_train):
    A=[[0,0],[0,0]]
    mew=Mean(Class_train)
    for i in range(len(Class_train[0])):
        for j in range(len(Class_train[0])):
            A[i][j]=get_Cov(Class_train,mew[i],mew[j],i,j)
    return A

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.
    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.
    """
    n = len(mu)
    Sigma_det = det=(Sigma[0][0]*Sigma[1][1])-(Sigma[1][0]*Sigma[0][1])
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-1*fac / 2) / N

def plot_contour(mu,Sigma,x,y):
    min_x=100000000
    min_y=100000000
    max_x=-100000000
    max_y=-100000000
    N = len(x)
    for i in range(len(x)):
        min_x=min(min_x,x[i])
        min_y=min(min_y,y[i])
        max_x=max(max_x,x[i])
        max_y=max(max_y,y[i])
    # print min_x,max_x,min_y,max_y
    X=np.linspace(min_x-3,max_x+3,N)
    Y=np.linspace(min_y-3,max_y+3,N)
    X, Y = np.meshgrid(X, Y)
    # Mean vector and covariance matrix
    # mu = np.array([0., 1.])
    # Sigma = np.array([[ 1. , -0.5], [-0.5,  1.5]])
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)
    # Create a surface plot and projected filled contour plot under it.
    plt.contour(X, Y, Z, colors='black',zorder=100,alpha=0.5)   

def plot(Class_train,color,label="",cont=False,mu=[],Sigma=[]):
    A=[]
    B=[]	
    for i in Class_train:
        A.append(i[0])
        B.append(i[1])
    if(label):
        plt.plot(A,B,color,markersize=3, label=label)
    elif(cont==True):
        plt.plot(A,B,color,markersize=3, label=label)
    else:
        plt.plot(A,B,color)
    if(cont==True):
        plot_contour(mu,Sigma,A,B)
    leg=plt.legend()
    for line in leg.get_lines():
        line.set_linewidth(4.0) 

def calc_Gx_case1(x,mean,var, prior):
    g_x = np.zeros(3, dtype=float)
    for i in range(3):
        arr = np.array([x-mean[i]])
        arrn = np.matmul(arr, arr.T)
        g_x[i]=-1*arrn[0][0]/(2*var) + np.log(prior[i])
    Max=-100000000000.0
    index=-1
    for i in range(3):
        if(Max<g_x[i]):
            Max = g_x[i]
            index=i
    return index

def calc_Gx_case2(x,mean,cov_mat,prior):
    g_x = np.zeros(3, dtype=float)
    for i in range(3):
        arr = np.array([x-mean[i]])
        arrt = arr.T
        cov_inv = np.linalg.inv(cov_mat) 
        arr1 = np.matmul(arr, cov_inv)
        arr2 = np.matmul(arr1,arrt)
        g_x[i] = -1*0.5*arr2[0][0] + np.log(prior[i])
    Max=-100000000000.0
    index=-1
    for i in range(3):
        if(Max<g_x[i]):
            Max = g_x[i]
            index=i
    return index

def calc_Gx_case3(x,mean,cov_mats,prior):
    g_x = np.zeros(3, dtype=float)
    for i in range(3):
        arr = np.array([x-mean[i]])
        arrt = arr.T
        cov = np.array(cov_mats[i])
        cov_inv = np.linalg.inv(cov) 
        arr1 = np.matmul(arr, cov_inv)
        arr2 = np.matmul(arr1,arrt)
        g_x[i] = -1*0.5*arr2[0][0] + np.log(prior[i]) - 0.5*np.log(np.linalg.det(cov))
    Max=-100000000000.0
    index=-1
    for i in range(3):
        if(Max<g_x[i]):
            Max = g_x[i]
            index=i
    return index

def Gx_case1_pair(x,mean1, mean2, var, p1,p2): 
    g_x = np.zeros(2, dtype=float)
    arr = np.array([x-mean1])
    arrn = np.matmul(arr, arr.T)
    g_x[0]=-1*arrn[0][0]/(2*var) + np.log(p1)
    arr = np.array([x-mean2])
    arrn = np.matmul(arr, arr.T)
    g_x[1]=-1*arrn[0][0]/(2*var) + np.log(p2)
    if(g_x[1]>g_x[0]):
        return 1
    else:
        return 0

def Gx_case2_pair(x,mean1, mean2, cov_mat, p1,p2):
    g_x = np.zeros(2, dtype=float)
    arr = np.array([x-mean1])
    cov_inv = np.linalg.inv(cov_mat) 
    arr1 = np.matmul(arr, cov_inv)
    arr2 = np.matmul(arr1,arr.T)
    g_x[0] = -1*0.5*arr2[0][0] + np.log(p1)
    arr = np.array([x-mean2])
    arr1 = np.matmul(arr, cov_inv)
    arr2 = np.matmul(arr1,arr.T)
    g_x[1] = -1*0.5*arr2[0][0] + np.log(p2)
    if(g_x[1]>g_x[0]):
        return 1
    else:
        return 0

def Gx_case3_pair(x,mean1, mean2, cov1,cov2, p1,p2):
    g_x = np.zeros(2, dtype=float)
    arr = np.array([x-mean1])
    cov_inv = np.linalg.inv(cov1)
    arr1 = np.matmul(arr, cov_inv)
    arr2 = np.matmul(arr1,arr.T)
    g_x[0] = -1*0.5*arr2[0][0] + np.log(p1) - 0.5*np.log(np.linalg.det(cov1))
    arr = np.array([x-mean2])
    cov_inv = np.linalg.inv(cov2)
    arr1 = np.matmul(arr, cov_inv)
    arr2 = np.matmul(arr1,arr.T)
    g_x[1] = -1*0.5*arr2[0][0] + np.log(p2) - 0.5*np.log(np.linalg.det(cov2))
    if(g_x[1]>g_x[0]):
        return 1
    else:
        return 0

def get_Score(Conf_Matrix):
	total=0.0
	True_val=0.0
	for i in range(len(Conf_Matrix)):
		for j in range(len(Conf_Matrix)):
			if(i==j):
				True_val=True_val+Conf_Matrix[i][j]
			total=total+Conf_Matrix[i][j]
	Accuracy=True_val/total
	Recall=[]
	Precision=[]
	for i in range(len(Conf_Matrix)):
		Sum=0.0
		for j in range(len(Conf_Matrix)):
			Sum=Sum+Conf_Matrix[i][j]
		Recall.append(Conf_Matrix[i][i]/Sum)
	for i in range(len(Conf_Matrix)):
		Sum=0.0
		for j in range(len(Conf_Matrix)):
			Sum=Sum+Conf_Matrix[j][i]
		if(Sum==0):
			Precision.append(0)
		else:
			Precision.append(Conf_Matrix[i][i]/Sum)
	print ("Accuracy of Classifier:- ",Accuracy)
	for i in range(len(Conf_Matrix)):
		print("Precision of Class",(i+1),":-",Precision[i])
	for i in range(len(Conf_Matrix)):
		print("Recall of Class",(i+1),":-",Recall[i])
	Sum=0.0
	for i in range(len(Conf_Matrix)):
		if ((Recall[i]+Precision[i]) == 0):
			print("F Measure of Class",(i+1),":- 0")
		else:
			print("F Measure of Class",(i+1),":-",(2*Recall[i]*Precision[i])/(Recall[i]+Precision[i]))
			Sum=Sum+(Recall[i]*Precision[i])/(Recall[i]+Precision[i])
	print("Mean Precision :-",(sum(Precision)/len(Conf_Matrix)))	
	print("Mean Recall :-",(sum(Recall)/len(Conf_Matrix)))
	print("Mean F Measure :-",(Sum)/len(Conf_Matrix)) 