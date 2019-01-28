# ECE521 A3
# March 18th 2017
# Wei Cui Leon Chen
# Question 1.1 part (2)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# parameters
K = 5
feature_amount = 2
epoch_amount = 400
batch_size = 500
# hyper-parameters
learning_rate = 0.001

# Tensorflow graph
data = tf.placeholder(tf.float32, shape=[None, feature_amount], name="data_placeholder")
with tf.name_scope("2D_mean"):
    mean = tf.Variable(tf.random_normal([K, feature_amount]), name="mean")
# compute the loss given the mean
sub_array = []
for i in range(K):
    sub_array_per_class = tf.reduce_sum(tf.pow((data - mean[i]), 2), axis=1)
    sub_array.append(sub_array_per_class)
assignments = tf.argmin(sub_array, axis=0)
sub_array = tf.reduce_min(sub_array, axis=0)
loss = tf.reduce_sum(sub_array)
optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-5)
train_step = optimizer.minimize(loss)


if(__name__ == '__main__'):
    # load dataset from data2D.npy
    data2D = np.load('data2D.npy') #should be a 2D array with size 10000 X 2
    orig_data = np.copy(data2D)
    n_val = data2D.shape[0]
    # parse data into minibatches
    # assuming batch size is divisible by the amount of data
    data2D = np.reshape(data2D,[-1, batch_size, feature_amount])

    batch_amount = np.shape(data2D)[0]

    total_loss = np.zeros(epoch_amount)
    centroids = np.zeros(K)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch_amount):
            epoch_loss = 0
            assign = []
            for j in range(batch_amount):
                centroids, batch_loss, assign_batch, _ = sess.run([mean, loss, assignments, train_step], feed_dict={data: data2D[j]})
                epoch_loss += batch_loss
                total_loss[i] = epoch_loss
                assign.append(assign_batch)
            if((i+1) % 20 == 0):
                print("At epoch{}, loss across all data: {}".format(i, epoch_loss))
        print("Model training complete!")
        save_path = saver.save(sess, "trained_model/model_1_1_3.ckpt")
        print("Trained model saved in file: {}".format(save_path))

    print("Script completed successfully!")

    # Percentage assigned to each cluster
    for cluster in range(K):
        print('{}% assigned to cluster {}'.format(round(np.mean(np.array(assign) == cluster)*100, 2), cluster+1))

    # Plot the data
    plt.figure(1)
    plt.scatter(orig_data[:, 0], orig_data[:, 1], c=assign, s=1)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='black', s=100)
    plt.title('K = {}'.format(K))
    plt.show()
