# Wei Cui Leon Chen
# March 24th 2017
# ECE521 A3 Question 3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# hyper-parameters
LEARNING_RATE = 0.001
EPOCH_AMOUNT = 1000
# parameters
minibatch_size = 100
D = 64
K = 4
# constant
PI = np.pi

# helper function to get the marginal log likelihood which we want to maximize
def marginal_log_likelihood(data, mean, weight, cov):
    cov_mat = tf.matmul(weight, tf.transpose(weight, [1,0])) + tf.diag(cov)
    log_coefficient = -D/2*np.log(2*PI)
    log_det_term = -tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(cov_mat))))
    log_likelihood_set = 0
    # loop over images. Don't think there's a vectorized approach of coding
    for i in range(minibatch_size):
        diff_mat = tf.expand_dims(data[i]-mean, axis=0)
        log_exp_term = -1/2*tf.matmul(tf.matmul(diff_mat, tf.matrix_inverse(cov_mat)), tf.transpose(diff_mat, [1,0]))
        log_likelihood_point = log_coefficient + log_det_term + log_exp_term
        log_likelihood_set += log_likelihood_point
    return log_likelihood_set

# placeholders
x = tf.placeholder(tf.float32, [None, 64], "data_placeholder")

with tf.name_scope("variables"):
    weight = tf.Variable(tf.truncated_normal(shape=[D, K], stddev=0.01, dtype=tf.float32, name='weight'))
    cov = tf.Variable(tf.truncated_normal(shape=[D], stddev=0.01, dtype=tf.float32, name="covariance"))
# mean
mean = tf.reduce_mean(x, axis=0) # 1XD
marginal_log_likelihood_value = marginal_log_likelihood(x, mean, weight, tf.exp(cov))
loss = -marginal_log_likelihood_value
optimizer = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.9, beta2=0.99, epsilon=1e-5)
train_step = optimizer.minimize(loss)

if(__name__=='__main__'):
    print("Loading the data")
    with np.load("tinymnist.npz") as data_dict:
        # image data pixel values are already normalized between 0 and 1
        train_data = data_dict["x"] # train_data: 700X64
        valid_data = data_dict["x_valid"] # valid_data: 100X64
        test_data = data_dict["x_test"] # test_data: 400X64
        np.random.seed(521)
        randIndx = np.arange(len(train_data))
        np.random.shuffle(randIndx)
        train_data = train_data[randIndx]
    print("Done loading the data!")

    # parse the train data into minibatches
    train_data_batches = np.reshape(train_data, [-1, minibatch_size, 64])
    train_batch_amount = np.shape(train_data_batches)[0]

    saver = tf.train.Saver()
    loss_epoches = []
    print("Starting tensorflow session...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(EPOCH_AMOUNT):
            epoch_loss = 0
            for j in range(train_batch_amount):
                batch_loss, _ = sess.run([loss, train_step], feed_dict={x: train_data_batches[j]})
                epoch_loss += batch_loss
            loss_epoches.append(epoch_loss)
            if((i+1) % 20 == 0):
                print("At epoch {}, loss across all training data: {}".format(i+1, epoch_loss[0][0]))
        print("Model training complete!")
        save_path = saver.save(sess, "trained_model/model_3_2.ckpt")
        print("Trained model saved in file: {}".format(save_path))
        # getting weights
        weights = sess.run(weight)

        # Validation
        valid_loss = sess.run(loss, feed_dict={x: valid_data})
        print('Validation Loss is: ', valid_loss[0][0])

        # Testing
        test_loss = sess.run(loss, feed_dict={x: test_data})
        print('Test Loss is: ', test_loss[0][0])

    print("saving trained weights...")
    np.save("trained_model/weights_3_2.npy", weights)
    print("Plotting weights:")
    weights = np.transpose(weights, [1,0])
    weights_plot = np.reshape(weights, [-1, 8, 8])
    fig = plt.figure(figsize=(2,2))
    for img in range(np.shape(weights_plot)[0]):
        ax = fig.add_subplot(2,2,img+1)
        ax.imshow(weights_plot[img], cmap='gray')
        ax.axis('off')
    fig.subplots_adjust(wspace = 0, hspace = 0)
    plt.show()
