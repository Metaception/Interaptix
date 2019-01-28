# ECE521 A3
# March 23th 2017
# Wei Cui Leon Chen
# Question 2.1 and 2.2

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *

# parameters
MINIBATCH_SIZE = 303
# hyper-parameters
LEARNING_RATE = 0.001
# constants
PI = np.pi
K = 1
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
    return log_marginal_likelihood_all, log_prob_posterior_unnormalized

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

loss, indiv_loss = log_marginal_likelihood(data, mean, std_vector, pi_vector)
loss = -loss
indiv_loss = -indiv_loss
assignments = tf.argmin(indiv_loss, axis=1)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.9, beta2=0.99, epsilon=1e-5)
train_step = optimizer.minimize(loss)

if(__name__ == '__main__'):
    # load dataset from data2D.npy
    data2D = np.load('data2D.npy') #should be a 2D array with size 10000 X 2
    tr_data = data2D[:int(data2D.shape[0]*2/3)]
    va_data = data2D[int(data2D.shape[0]*2/3):]
    n_tr = tr_data.shape[0]
    n_va = va_data.shape[0]
    # parse data into minibatches
    # assuming batch size is divisible by the amount of data
    tr_data = np.reshape(tr_data, [-1, MINIBATCH_SIZE, FEATURE_AMOUNT])
    batch_amount = np.shape(tr_data)[0]

    saver = tf.train.Saver()
    loss_epoches = []
    print("Starting tensorflow session...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(EPOCH_AMOUNT):
            epoch_loss = 0
            for j in range(batch_amount):
                batch_loss, _ = sess.run([loss, train_step], feed_dict={data: tr_data[j]})
                epoch_loss += batch_loss
            loss_epoches.append(epoch_loss)
            if((i+1) % 25 == 0):
                print("At epoch {}, loss across all data: {}".format(i+1, epoch_loss))
        print("Model training complete!")
        save_path = saver.save(sess, "trained_model/model_2_2.ckpt")
        print("Trained model saved in file: {}".format(save_path))

        # Validation
        valid_loss = sess.run(loss, feed_dict={data: va_data})
        print('Validation Loss is: ', valid_loss)

        # Final cluster
        centroids, assign = sess.run([mean, assignments], feed_dict={data: data2D})

        # Percentage assigned to each cluster
        for cluster in range(K):
            print('{}% assigned to cluster {}'.format(round(np.mean(np.array(assign) == cluster)*100, 2), cluster+1))

    print("Plotting losses vs number of updates:")
    # Plot the data
    plt.figure(1)
    plt.scatter(data2D[:, 0], data2D[:, 1], c=assign, s=1)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='black', s=100)
    plt.title('K = {}'.format(K))
    plt.show()
    print("Script completed successfully!")
