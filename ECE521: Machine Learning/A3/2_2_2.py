# ECE521 A3
# March 23th 2017
# Wei Cui Leon Chen
# Question 2.1 and 2.2

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *

# parameters
MINIBATCH_SIZE = 500
# hyper-parameters
LEARNING_RATE = 0.001
# constants
PI = np.pi
K = 3
FEATURE_AMOUNT = 2
EPOCH_AMOUNT = 500

# For 2.1
def log_likelihood(data, mean, stddev_vector):
    # data: BXD; mean: KXD; variance: K (given uni-standard deviation assumption)
    log_probs = []
    for i in range(K):
        dists = data - mean[i] # through broadcasting, shape is BXD
        gaussian_coefficient = -FEATURE_AMOUNT/2*tf.log(2*PI) - FEATURE_AMOUNT*tf.log(stddev_vector[0,i])
        # Taking the log should turn into summation terms; result is with dimension: BX1
        log_prob_per_class = gaussian_coefficient + (-1/(2*tf.pow(stddev_vector[0,i],2))*tf.reduce_sum(tf.pow(dists,2),axis=1))
        log_probs.append(log_prob_per_class) # shape: KXB
    # result takes on the shape: KXB, transform it for more natural representation
    log_probs = tf.transpose(log_probs, [1,0]) # BXK
    return log_probs

# For 2.1
def log_posterior(data, mean, stddev_vector, pi_vector):
    # assume pi_vector: 1XK
    # compute the log probability for "cluster" specified by the parameter
    # compute term for each cluster (p(x|z) added with a prior)
    log_joint_prob = []
    log_prob_likelihood = log_likelihood(data, mean, stddev_vector) # BXK
    # compute the log prior
    log_prob_prior = pi_vector  # tf.log(pi_vector)
    log_prob_posterior_unnormalized = log_prob_prior + log_prob_likelihood # BXK
    # now ready to compute the probability on posterior for each cluster
    log_prob_posterior = log_prob_posterior_unnormalized / reduce_logsumexp(log_prob_posterior_unnormalized, reduction_indices=1) # BXK
    return log_prob_posterior

# For 2.2
def log_marginal_likelihood(data, mean, stddev_vector, pi_vector):
    # compute term for each cluster (p(x|z) added with a prior)
    log_joint_prob = []
    log_prob_likelihood = log_likelihood(data, mean, stddev_vector) # BXK
    # compute the log prior
    log_prob_prior = pi_vector  # tf.log(pi_vector)
    log_prob_posterior_unnormalized = log_prob_prior + log_prob_likelihood # BXK
    log_marginal_likelihood_per_point = reduce_logsumexp(log_prob_posterior_unnormalized, reduction_indices=1) # BX1
    log_marginal_likelihood_all = tf.reduce_sum(log_marginal_likelihood_per_point)
    return log_marginal_likelihood_all

# placeholders
data = tf.placeholder(tf.float32, shape=[None, FEATURE_AMOUNT], name="data_placeholder")
with tf.name_scope("MoG_parameters"):
    mean = tf.Variable(tf.random_normal([K, FEATURE_AMOUNT]), name="mean")
    # for parametrizing standard deviation
    var_parametrize = tf.Variable(tf.random_normal([1,K]), name='var_parametrize')
    # for parametrizing pi
    pi_parametrize = tf.Variable(tf.random_normal([1,K]), name='pi_parametrize')

# get variance vector and pi vector
var_vector = tf.exp(var_parametrize); std_vector = tf.sqrt(var_vector)
pi_vector =  logsoftmax(pi_parametrize)

loss = -log_marginal_likelihood(data, mean, std_vector, pi_vector)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.9, beta2=0.99, epsilon=1e-5)
train_step = optimizer.minimize(loss)

if(__name__ == '__main__'):
    # load dataset from data2D.npy
    data2D = np.load('data2D.npy') #should be a 2D array with size 10000 X 2
    # parse data into minibatches
    # assuming batch size is divisible by the amount of data
    data2D = np.reshape(data2D,[-1, MINIBATCH_SIZE, FEATURE_AMOUNT])

    batch_amount = np.shape(data2D)[0]
    saver = tf.train.Saver()
    loss_epoches = []
    print("Starting tensorflow session...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(EPOCH_AMOUNT):
            epoch_loss = 0
            for j in range(batch_amount):
                batch_loss, centroids, pi_log, sigma, _ = sess.run([loss, mean, pi_vector, std_vector, train_step], feed_dict={data: data2D[j]})
                epoch_loss += batch_loss
            loss_epoches.append(epoch_loss)
            if((i+1) % 25 == 0):
                print("At epoch {}, loss across all data: {}".format(i+1, epoch_loss))
        print("Model training complete!")
        save_path = saver.save(sess, "trained_model/model_2_2.ckpt")
        print("Trained model saved in file: {}".format(save_path))

        print("Mean: {}".format(centroids))
        print("Pi: {}".format(np.exp(pi_log)))
        print("Sigma: {}".format(sigma))

    print("Plotting losses vs number of updates:")
    plt.title("Loss VS Number of Updates")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log Marginal Likelihood")
    plt.plot(loss_epoches)
    plt.show()
    print("Script completed successfully!")
