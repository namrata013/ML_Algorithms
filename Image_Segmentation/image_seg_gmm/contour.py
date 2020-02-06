from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import statistics_func as sf
# Our 2-dimensional distribution will be over variables X and Y
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
