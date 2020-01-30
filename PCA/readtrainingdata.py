from mnist import MNIST
from array import *
from math import *
import random
import numpy 
import matplotlib.pyplot as plt
import sys
from pathlib import Path

#dimensions/featurs = 28x28 = 784
dimension = 784
def readata():
    f = Path("data/")
    ff = f / ""
    # mndata = MNIST('/home/namrata/Desktop/sem5/pr/ass3')
    mndata = MNIST(ff)
    images, labels = mndata.load_training()
    #  or
    #  images, labels = mndata.load_testing()
    # return images, labels

    index = random.randrange(0, len(images))  # choose an index ;-)
    print(mndata.display(images[index]))
    print(labels[index])
    print(len(images))  
    #  60000
    print(len(labels)) 
    #  60000
    len(images[0]) 
    # 784 
    #  (28x28 image as a 1-d vector)
    #  number of features will be the same as the number of pixels=784
    return images, labels

def covarianceMat(images,labels,label):
    i=0
    labelimages = []
    # j=0
    for i in range(len(labels)):
        if(labels[i]==label):
            labelimages.append(images[i])
    return numpy.cov(labelimages)
    #         j=j+1
    #         if(j==1):
    #             break
    # print(labelimages)

def main():
    images, labels = readata()
    label=6
    covarianceMat(images,labels,label)

if __name__== "__main__":
    main()



