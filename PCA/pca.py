import sys
from pathlib import Path
import numpy as np
import math
import struct
from scipy import linalg as la
import matplotlib.pyplot as plt
from mnist import MNIST


def computeMean(data):
    row = data.shape[0]
    col = data.shape[1]
    mean = np.zeros((row,),dtype=float)
    for c in range(col):
        for r in range(row):
            mean[r] += float(data[r][c]/col)
    return mean

def dataSelection(data, select, labels):
    row = data.shape[0]
    col = data.shape[1]
    N = 0
    for c in range(col):
        if labels[c]==select:
            N += 1
    data_select = np.zeros((row,N), dtype=float)
    index = 0
    for c in range(col):
        if labels[c]==select:
            for r in range(row):
                data_select[r][index] += float(data[r][c])
            index += 1
    return data_select

def meanNormalisation(data, mean):
    row = data.shape[0]
    col = data.shape[1]
    for c in range(col):
        for r in range(row):
            data[r][c] = float(data[r][c]-mean[r])
    return data

def computeCovariance(data):
    col = data.shape[1]
    covariance = np.matmul(data,np.transpose(data))/col
    return covariance

def reconstruct(X, U):
    # d = 784
    # Y(K*n) = U(d*K)_transpose X(d*n)
    # X(d*n) = U(d*K) Y(K*n)
    Y = np.matmul(np.transpose(U),X)
    X = np.matmul(U,Y)

    print("dim X  : ",X.shape)
    print("dim U  : ",U.shape)
    print("dim Y  : ",Y.shape)
    return X

def diff(data_1, data_2):
    row = data_1.shape[0]
    col = data_1.shape[1]
    diff = float(0)
    for r in range(row):
        for c in range(col):
            diff += abs(data_1[r][c]-data_2[r][c])
    return diff/(row*col)

def main():
    print()
    # read images from dataset
    folder = Path("data/")
    file1 = folder / "train-images-idx3-ubyte"
    with open(file1,'rb') as f:
        magic_number, size = struct.unpack(">II", f.read(8))
        num_row, num_col = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size, num_row, num_col))
    print("Dim Data   : ",data.shape)
    # make X = d*N
    # dimensions d, total number of samples N
    data = data.reshape(data.shape[0],data.shape[1]*data.shape[2])
    print("Dim Data   : ",data.shape)
    data = np.transpose(data)
    print("Dim Data   : ",data.shape)

    # read labels from dataset
    file2 = folder / "train-labels-idx1-ubyte"
    with open(file2,'rb') as f:
        magic_number, size = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    print("Dim Labels : ",labels.shape)

    print("\n-------------------- ----------------------\n")
    print("Enter digit (0-9)")
    select = int(input())
    print("\n-------------------- ----------------------\n")
    # select all samples of given digit
    data_select = dataSelection(data, select, labels)
    print("Dim Data   : ",data_select.shape)
    # calculate mean
    mean = computeMean(data_select)
    print("Dim Mean   : ",mean.shape)
    # normalise the data by subtracting mean from all samples
    data_normalised = meanNormalisation(data_select, mean)
    # calculate covariance of data
    covariance = computeCovariance(data_normalised)
    print("Dim Cov    : ",covariance.shape)

    print("\n-------------------- ----------------------\n")
    # calculate eigen values and eigen vectors of covariance matrix
    eigenValue,eigenVector = la.eigh(covariance)
    print("Dim Eigen Value  :",eigenValue.shape)
    print("Dim Eigen Vector :",eigenVector.shape)

    # sort eigen values and return the indices that sort it
    sortedIndex = np.argsort(eigenValue)[::-1]
    eigenValue = eigenValue[sortedIndex]
    # arrange eigen vectors coressponding to sorted eigen values column wise
    eigenVector = eigenVector[:,sortedIndex]

    # calculate energy of all eigen vectors
    energy_original = float(0)
    row = eigenVector.shape[0]
    col = eigenVector.shape[1]
    for r in range(row):
        for c in range(col):
            energy_original += (eigenVector[r][c])**2

    print("\n-------------------- ----------------------\n")
    print("Enter K ( 1-",eigenVector.shape[1],")\n(-1) if find optimal K")
    K_input = int(input())
    if(K_input==-1):
        print("\nEnter threshold (0-100)")
        thresh = float(input())
        # loop over k = 1 to K to find optimal K
        energy = float(0)
        for k in range(0,col):
            # calculate energy of k eigen vectors
            for r in range(row):
                energy += (eigenVector[r][k])**2
            if (energy/energy_original)>=thresh/100:
                break

    if K_input==-1:
        K = k+1
    else:
        K = K_input

    print("\n-------------------- ----------------------\n")

    # choose K eigen vectors
    eigenValue_select = eigenValue[:K]
    print("Dim Eigen Value  :", eigenValue_select.shape)
    eigenVector_select = eigenVector[:, :K]
    print("Dim Eigen Vector :", eigenVector_select.shape)
    print()

    # reconstruct image
    data_reconstructed = reconstruct(data_select, eigenVector_select)
    print("Dim Recon  :",data_reconstructed.shape)

    if K_input==-1:
        print("\n-------------------- ----------------------\n")
        print("Energy Fraction Threshold :",thresh," %")
        print("Minimum K                 :",K,"/",eigenVector.shape[0])
        print()
    else:
        energy = float(0)
        for k in range(0,K):
            for r in range(row):
                energy += (eigenVector[r][k])**2
        fraction = (energy/energy_original)*100
        print("\n-------------------- ----------------------\n")
        print("Input K         :",K,"/",eigenVector.shape[0])
        print("Energy Fraction :","%.3f" %fraction,"%")
        print("\n-------------------- ----------------------\n")
        print("Avg pixel value diff ORIGINAL vs FINAL :", diff(data_select, data_reconstructed))
        print()

    # reshape image
    orig = np.transpose(data_select)
    orig = orig.reshape((orig.shape[0],int(math.sqrt(orig.shape[1])),int(math.sqrt(orig.shape[1]))))
    recon = np.transpose(data_reconstructed)
    recon = recon.reshape((recon.shape[0],int(math.sqrt(recon.shape[1])),int(math.sqrt(recon.shape[1]))))
    # plot image
    plt.imshow(orig[0,:,:], cmap='gray')
    plt.show()
    plt.imshow(recon[0,:,:], cmap='gray')
    plt.show()

if __name__=="__main__":
    main()
