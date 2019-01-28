# ECE521 A3
# March 23th 2017
# Wei Cui Leon Chen
# Question 3.1.3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# PCA and FA from Scikit-Learn
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis

if(__name__ == '__main__'):
    mean =  [0, 0, 0]
    covar = np.eye(3)
    latent = np.random.multivariate_normal(mean, covar, 200)            # s = N(s; 0, I)
    data = np.matmul([[1, 0, 0], [1, 0.1, 0], [0, 0, 10]], latent.T)    # x = [s1, s1 + 0.1*s2, s3]
    data = data.T

    # PCA
    pca = PCA(n_components=1).fit(data)
    print('PCA: ', pca.components_)

    # Factor Analysis
    factor =    FactorAnalysis(n_components=1).fit(data)

    # Get W and Psi inverse
    W =     factor.components_.T
    info =  np.linalg.inv(factor.get_covariance())

    # Wproj = ((I + W^T * Psi^-1 * W)^-1 * W)^T * Psi^-1
    sigma = covar + np.matmul(np.matmul(W.T, info), W)
    sigma = np.linalg.inv(sigma)
    Wproj = np.matmul(np.matmul(sigma, W).T, info)
    print('FA:  ', Wproj)
