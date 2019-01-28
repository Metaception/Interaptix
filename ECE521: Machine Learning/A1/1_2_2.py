# Question 1.2 Part 2
import tensorflow as tf
import numpy as np

def squared_dist(a, b):
	# To broadcast, we want the first matrix to be with shape Bx1xN
	# while the second matrix to be with shape 1xCxN
	a_expand =  tf.expand_dims(a, 1)
	b_expand =  tf.expand_dims(b, 0)

	# Only summing over the third dimension, while preserving the first two dimensions
	dist =  tf.reduce_sum(tf.squared_difference(a_expand, b_expand), axis=2)
	return dist


# Establish the main function to test and verify the code
if __name__ == '__main__':
	# amount of data points for first input matrix
	B = 10
	# amount of data points for second input matrix
	C = 5
	# amount of features per data point
	N = 3

	a = tf.placeholder('float', [B, N]) # could use None but enforce row checking
	b = tf.placeholder('float', [C, N]) # could use None but enforce row checking

	# Generate two inputs
	A = np.random.rand(B, N)
	B = np.random.rand(C, N)

	# Squared Euclidean Distance
	dist =  squared_dist(a, b)

	with tf.Session() as sess:
		result = sess.run(dist, feed_dict={a: A, b: B})
		print("The resulting matrix (expecting dimension of BxC):\n {}".format(result))
