from __future__ import print_function
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math
import matplotlib.mlab as mlab
fig, (ax) = plt.subplots(ncols=1)
x=np.linspace(-3000,3000) 
plt.axis('equal')

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
    return np.exp(-fac / 2) / N

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
    Z = multivariate_gaussian(pos, mu, Sigma)
    # The distribution on the variables X, Y packed into pos.
    # Create a surface plot and projected filled contour plot under it.
    plt.contour(X, Y, Z, cmap="RdBu_r",zorder=100,alpha=0.5)
    # print("hello")    


def plot(Class_train,color,label="",cont=False,mu=[],Sigma=[]):
	A=[]
	B=[]	
	c = 1
	for i in Class_train:
		A.append(i[0])
		B.append(i[1])
		c+=1
	if(label):
		plt.plot(A,B,color,markersize=3, label=label)
	elif(cont==True):
		plt.plot(A,B,color,markersize=3, label=label)
	else:
		plt.plot(A,B,color)
	if(cont==True):
		plot_contour(mu,Sigma,A,B)
	plt.legend()

def get_data(file):
	train=[]
	test=[]
	fo=open(file,"r")
	X=[]
	for line in fo:
		a,b=line.split()
		X.append([float(a),float(b)])
	# random.shuffle(X) #randomly divide the dataset into 75% training and 25%test
	train=X[:int(len(X)*(0.75))]
	test=X[int(len(X)*0.75):]
	fo.close()
	return train,test

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
	print("Mean F Measure :-",2*(Sum)/len(Conf_Matrix))
	# print("PLZZ check formula for F measure before reporting")