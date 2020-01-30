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
    cov_mat1 = np.zeros((2,2),dtype=float)
    cov_mat2 = np.zeros((2,2),dtype=float)
    cov_mat3 = np.zeros((2,2),dtype=float)
    mean_mat = []
    Data = []
    prior = np.zeros(3)
    RANGE = [[0,0],[0,0]]
    step=1
    def __init__(self,DATASET,tmp):
        self.DATA = DATASET
        for i in range(len(DATASET)):
            self.mean_mat.append(sf.Mean(DATASET[i]))
        self.mean_mat = np.array(self.mean_mat)
        self.cov_mat1 = sf.cov_mat(DATASET[0])
        self.cov_mat2 = sf.cov_mat(DATASET[1])
        self.cov_mat3 = sf.cov_mat(DATASET[2]) 
        for i in range(3):
            self.prior[i]= len(DATASET[i])/(len(DATASET[0])+len(DATASET[1])+len(DATASET[2]))
        if(tmp==0):
            for i in range(2):
                for j in range(2):
                    if i!=j :
                        self.cov_mat1[i][j] = 0
                        self.cov_mat2[i][j] = 0
                        self.cov_mat3[i][j] = 0        
    def Plot_Classifier(self,data_id):
        self.step = 1
        self.RANGE[0][0], self.RANGE[0][1], self.RANGE[1][0], self.RANGE[1][1] = 0,0,0,0
        if (data_id == 1):
            self.step = 0.2
            self.RANGE[0][0], self.RANGE[0][1], self.RANGE[1][0], self.RANGE[1][1] = -10,25,-20,20
        elif( data_id == 2):
            self.step = 0.05
            self.RANGE[0][0], self.RANGE[0][1], self.RANGE[1][0], self.RANGE[1][1] = -4,4,-4,4
        elif( data_id == 3):
            self.step = 20
            self.RANGE[0][0], self.RANGE[0][1], self.RANGE[1][0], self.RANGE[1][1] = 0,1000,0,2500
        temp=[[],[],[]]
        cov_mats =[self.cov_mat1, self.cov_mat2, self.cov_mat3]
        i=self.RANGE[0][0]
        while i<=self.RANGE[0][1]:
            j=self.RANGE[1][0]
            while j<=self.RANGE[1][1]:
                index=-1
                x = np.array([i,j])
                index = sf.calc_Gx_case3(x,self.mean_mat, cov_mats, self.prior)
                temp[index].append([i,j])
                j=j+self.step
            i=i+self.step
        sf.plot(temp[0],'b',"Class1")
        sf.plot(temp[1],'g',"Class2")
        sf.plot(temp[2],'r',"Class3")
        sf.plot(self.DATA[0],'mo',"",True,self.mean_mat[0],self.cov_mat1)
        sf.plot(self.DATA[1],'yo',"",True,self.mean_mat[1],self.cov_mat2)
        sf.plot(self.DATA[2],'co',"",True,self.mean_mat[2],self.cov_mat3)
        sf.plot([sf.Mean(self.DATA[0])],'ko')
        sf.plot([sf.Mean(self.DATA[1])],'ko')
        sf.plot([sf.Mean(self.DATA[2])],'ko')
        plt.legend()
        folder = Path("Results/")
        plt.savefig(folder / 'Classify_RealWorldData.png', dpi=300, bbox_inches='tight')
        plt.show()        
    def plot_pair(self):
        # class 1 and class 2
        mean1 = np.array(self.mean_mat[0])
        mean2 = np.array(self.mean_mat[1])
        p1 = self.prior[0]
        p2 = self.prior[1]
        temp=[[],[]]
        i=self.RANGE[0][0]
        while i<=self.RANGE[0][1]:
            j=self.RANGE[1][0]
            while j<=self.RANGE[1][1]:
                index=-1
                x = np.array([i,j])
                index = sf.Gx_case3_pair(x,mean1, mean2, self.cov_mat1, self.cov_mat2, p1,p2)
                temp[index].append([i,j])
                j=j+self.step
            i=i+self.step
        sf.plot(temp[0],'r',"Class1")
        sf.plot(temp[1],'b',"Class2")
        sf.plot(self.DATA[0],'go',"",True,self.mean_mat[0],self.cov_mat1)
        sf.plot(self.DATA[1],'yo',"",True,self.mean_mat[2],self.cov_mat2)
        plt.show()
        # class 2 and class 3
        mean1 = np.array(self.mean_mat[1])
        mean2 = np.array(self.mean_mat[2])
        p1 = self.prior[1]
        p2 = self.prior[2]
        temp=[[],[]]
        i=self.RANGE[0][0]
        while i<=self.RANGE[0][1]:
            j=self.RANGE[1][0]
            while j<=self.RANGE[1][1]:
                index=-1
                x = np.array([i,j])
                index = sf.Gx_case3_pair(x,mean1, mean2, self.cov_mat2, self.cov_mat3, p1,p2)
                temp[index].append([i,j])
                j=j+self.step
            i=i+self.step
        sf.plot(temp[0],'r',"Class2")
        sf.plot(temp[1],'b',"Class3")
        sf.plot(self.DATA[1],'go',"",True,self.mean_mat[1],self.cov_mat2)
        sf.plot(self.DATA[2],'yo',"",True,self.mean_mat[2],self.cov_mat3)
        plt.show()
        # class 2 and class 3
        mean1 = np.array(self.mean_mat[2])
        mean2 = np.array(self.mean_mat[0])
        p1 = self.prior[2]
        p2 = self.prior[0]
        temp=[[],[]]
        i=self.RANGE[0][0]
        while i<=self.RANGE[0][1]:
            j=self.RANGE[1][0]
            while j<=self.RANGE[1][1]:
                index=-1
                x = np.array([i,j])
                index = sf.Gx_case3_pair(x,mean1, mean2, self.cov_mat3, self.cov_mat1, p1,p2)
                temp[index].append([i,j])
                j=j+self.step
            i=i+self.step
        sf.plot(temp[0],'r',"Class3")
        sf.plot(temp[1],'b',"Class1")
        sf.plot(self.DATA[2],'go',"",True,self.mean_mat[2],self.cov_mat3)
        sf.plot(self.DATA[0],'yo',"",True,self.mean_mat[0],self.cov_mat1)
        plt.show()
    def conf_mat(self, TESTSET):
        CONF = [[0,0,0],[0,0,0],[0,0,0]]
        cov_mats =[self.cov_mat1, self.cov_mat2, self.cov_mat3]
        for i in range(len(TESTSET)):
            for j in range(len(TESTSET[i])):
                index=-1
                x = np.array([TESTSET[i][j][0],TESTSET[i][j][1]])
                index = sf.calc_Gx_case3(x,self.mean_mat,cov_mats, self.prior)
                CONF[i][index] = CONF[i][index]+1
        print("Confusion Matrix")
        for i in range(3):
            for j in range(3):
                print(CONF[i][j], end=" ")
            print("")
        sf.get_Score(CONF) 
                

        


        

