import sys
import numpy as np
import math
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import functions as sf
from pathlib import Path

# DATASET[i][j][k] 
# i class index
# j datapoint index
# k feature index

class Model():
    cov_mat = np.zeros((2,2),dtype=float)
    mean_mat = []
    Data = []
    prior = np.zeros(3)
    def __init__(self,DATASET):
        self.DATA = DATASET
        for i in range(len(DATASET)):
            self.mean_mat.append(sf.Mean(DATASET[i]))
        self.mean_mat = np.array(self.mean_mat)
        cov_mat_class1 = sf.cov_mat(DATASET[0])
        cov_mat_class2 = sf.cov_mat(DATASET[1])
        cov_mat_class3 = sf.cov_mat(DATASET[2]) 
        for i in range(3):
            self.prior[i]= len(DATASET[i])/(len(DATASET[0])+len(DATASET[1])+len(DATASET[2]))
        for i in range(2):
            for j in range(2):
                self.cov_mat[i][j] = cov_mat_class1[i][j] + cov_mat_class2[i][j] + cov_mat_class3[i][j]
        for i in range(2):
            for j in range(2):
                self.cov_mat[i][j] = self.cov_mat[i][j]/3
    def Plot_Classifier(self,RANGE,val):
        temp=[[],[],[]]
        i=RANGE[0][0]
        while i<=RANGE[0][1]:
            j=RANGE[1][0]
            while j<=RANGE[1][1]:
                index=-1
                x = np.array([i,j])
                index = sf.calc_Gx_case2(x,self.mean_mat,self.cov_mat, self.prior)
                temp[index].append([i,j])
                j=j+val
            i=i+val
        sf.plot(temp[0],'b',"Class1")
        sf.plot(temp[1],'g',"Class2")
        sf.plot(temp[2],'r',"Class3")
        sf.plot(self.DATA[0],'mo',"",True,self.mean_mat[0],self.cov_mat)
        sf.plot(self.DATA[1],'yo',"",True,self.mean_mat[1],self.cov_mat)
        sf.plot(self.DATA[2],'co',"",True,self.mean_mat[2],self.cov_mat)
        sf.plot([sf.Mean(self.DATA[0])],'ko')
        sf.plot([sf.Mean(self.DATA[1])],'ko')
        sf.plot([sf.Mean(self.DATA[2])],'ko')
        plt.legend()
        plt.show()
    def plot_pair(self,RANGE,val):
        # class 1 and class 2
        mean1 = np.array(self.mean_mat[0])
        mean2 = np.array(self.mean_mat[1])
        p1 = self.prior[0]
        p2 = self.prior[1]
        temp=[[],[]]
        i=RANGE[0][0]
        while i<=RANGE[0][1]:
            j=RANGE[1][0]
            while j<=RANGE[1][1]:
                index=-1
                x = np.array([i,j])
                index = sf.Gx_case2_pair(x,mean1, mean2, self.cov_mat, p1,p2)
                temp[index].append([i,j])
                j=j+val
            i=i+val
        sf.plot(temp[0],'r',"Class1")
        sf.plot(temp[1],'b',"Class2")
        sf.plot(self.DATA[0],'go',"",True,self.mean_mat[0],self.cov_mat)
        sf.plot(self.DATA[1],'yo',"",True,self.mean_mat[1],self.cov_mat)
        plt.show()
        # class 2 and class 3
        mean1 = np.array(self.mean_mat[1])
        mean2 = np.array(self.mean_mat[2])
        p1 = self.prior[1]
        p2 = self.prior[2]
        temp=[[],[]]
        i=RANGE[0][0]
        while i<=RANGE[0][1]:
            j=RANGE[1][0]
            while j<=RANGE[1][1]:
                index=-1
                x = np.array([i,j])
                index = sf.Gx_case2_pair(x,mean1, mean2, self.cov_mat, p1,p2)
                temp[index].append([i,j])
                j=j+val
            i=i+val
        sf.plot(temp[0],'r',"Class2")
        sf.plot(temp[1],'b',"Class3")
        sf.plot(self.DATA[1],'go',"",True,self.mean_mat[1],self.cov_mat)
        sf.plot(self.DATA[2],'yo',"",True,self.mean_mat[2],self.cov_mat)
        plt.show()
        # class 3 and class 1
        mean1 = np.array(self.mean_mat[2])
        mean2 = np.array(self.mean_mat[0])
        p1 = self.prior[2]
        p2 = self.prior[0]
        temp=[[],[]]
        i=RANGE[0][0]
        while i<=RANGE[0][1]:
            j=RANGE[1][0]
            while j<=RANGE[1][1]:
                index=-1
                x = np.array([i,j])
                index = sf.Gx_case2_pair(x,mean1, mean2, self.cov_mat, p1,p2)
                temp[index].append([i,j])
                j=j+val
            i=i+val
        sf.plot(temp[0],'r',"Class3")
        sf.plot(temp[1],'b',"Class1")
        sf.plot(self.DATA[2],'go',"",True,self.mean_mat[2],self.cov_mat)
        sf.plot(self.DATA[0],'yo',"",True,self.mean_mat[0],self.cov_mat)
        plt.show()
    def conf_mat(self, TESTSET):
        CONF = [[0,0,0],[0,0,0],[0,0,0]]
        for i in range(len(TESTSET)):
            for j in range(len(TESTSET[i])):
                index=-1
                x = np.array([TESTSET[i][j][0],TESTSET[i][j][1]])
                index = sf.calc_Gx_case2(x,self.mean_mat,self.cov_mat, self.prior)
                CONF[i][index] = CONF[i][index]+1
        print("Confusion Matrix")
        for i in range(3):
            for j in range(3):
                print(CONF[i][j], end=" ")
            print("")
        sf.get_Score(CONF)     
                

        



                

        


        

