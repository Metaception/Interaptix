# ECE521 A3
# March 18th 2017
# Wei Cui Leon Chen
# Question 2.1 Part (2)

import numpy as np
import tensorflow as tf    
from utils import *                                       

# parameters
# constants
PI = np.pi
def log_likelihood(data, mean, stddev):
	# data: BXD; mean: KXD; variance: K (given uni-standard deviation assumption)
	K, D = np.shape(mean)
	log_prob = []
	for i in range(K):
		dists = data - mean[i] # through broadcasting, shape is BXD
		gaussian_coefficient = -D/2*np.log(2*PI) - D*np.log(stddev[i])
		# Taking the log should turn into summation terms; result is with dimension: BX1
		log_prob_per_class = gaussian_coefficient + (-1/(2*np.power(stddev[i],2))*np.sum(np.power(dists,2),axis=1))
		log_prob.append(log_prob) # shape: KXB
	# result takes on the shape: KXB, transform it for more natural representation
	log_prob = np.transpose(log_prob, [1,0]) # BXK
	return log_prob

def log_prob_cluster(data, mean, stddev, cluster, pi_vector):
	# assume pi_vector: 1XK
	# compute the log probability for "cluster" specified by the parameter
    K, D = np.shape(mean)
    # compute term for each cluster (p(x|z) added with a prior)
    log_joint_prob = []
    log_likelihood_prob = log_likelihood(data, mean, stddev) # BXK
    # compute the log prior 
    log_prob_prior = np.log(pi_vector) 
    log_prob_posterior_unnormalized = log_prob_prior + log_prob_likelihood # BXK
    # now ready to compute the probability on posterior for each cluster
    log_prob_posterior = log_prob_posterior_unnormalized / reduce_logsumexp(log_prob_posterior_unnormalized, axis=1) # BXK
    return log_prob_posterior

if(__name__ == '__main__'):
    # load dataset from data2D.npy
    data2D = np.load('data2D.npy') #should be a 2D array with size 10000 X 2