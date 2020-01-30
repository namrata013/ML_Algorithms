import sys
import numpy as np
import Bayes_a
import Bayes_b
import Bayes_c
import functions as fs
from pathlib import Path
import matplotlib.pyplot as plt
No_classes=3
data_dimension=2
RANGE=[]
val=0.05

# DATASET[i][j][k] 
# i class index
# j datapoint index
# k feature index

data = input('Choose the type of your data \n choose 1 for Linearly separable \n 2 for Non Linearly Separable \n 3 for Real World Data \n ') 
while(int(data)>3 or int(data)<1):
    data = input("Try again : ")
d = int(data)

classifier = input('Choose the type of classifier \n choose 1 for same cov mat for all classes and equal to σ2I \n 2 for same cov mat for all classes and is Σ \n 3 for diagonal cov mat and different for all classes \n 4 for full cov mat and different for each class \n')
while int(classifier)>4 or int(classifier)<1 :
    classifier = input("try again: ")
c = int(classifier)

data_folder = " "
if d == 1:
    data_folder = Path("Data/linear_sep/")
    RANGE=[[-10,25],[-20,20]]
if d==2:
    data_folder = Path("Data/nonlinear_sep/")
    RANGE=[[-4,4],[-4,4]]
if d==3:
    RANGE=[[0,1000],[0,2500]]
    val=1
    data_folder = Path("Data/realworld_data/")

file_to_open = data_folder / "Class1.txt"
c1_train, c1_test = fs.getdata(file_to_open)
# print(c1_train) 
mean = fs.Mean(c1_train)
# print(mean)
# print(fs.variance(c1_train,mean))
file_to_open = data_folder / "Class2.txt"
c2_train, c2_test = fs.getdata(file_to_open)
file_to_open = data_folder / "Class3.txt"
c3_train, c3_test = fs.getdata(file_to_open)
DATASET = [c1_train, c2_train, c3_train]
TESTSET = [c1_test, c2_test, c3_test]
DATASET = np.array(DATASET)
TESTSET = np.array(TESTSET)

if c==1:
    model = Bayes_a.Model(DATASET)
    model.Plot_Classifier(RANGE,val)
    model.plot_pair(RANGE,val)
    model.conf_mat(TESTSET)
if c==2:
    model = Bayes_b.Model(DATASET)
    model.Plot_Classifier(RANGE,val)
    model.plot_pair(RANGE,val)
    model.conf_mat(TESTSET)
if c==3:
    tmp=0
    model = Bayes_c.Model(DATASET,tmp)
    model.Plot_Classifier(d)
    model.plot_pair()
    model.conf_mat(TESTSET)
if c==4:
    tmp=1
    model = Bayes_c.Model(DATASET,tmp)
    model.Plot_Classifier(d)
    model.plot_pair()
    model.conf_mat(TESTSET)
