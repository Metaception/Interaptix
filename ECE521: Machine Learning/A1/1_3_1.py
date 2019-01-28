# Question 1.3 Part 1
import numpy as np
import tensorflow as tf

def squared_dist(a, b):
	# To broadcast, we want the first matrix to be with shape Bx1xN
	# while the second matrix to be with shape 1xCxN
	a_expand =	tf.expand_dims(a, 1)
	b_expand =	tf.expand_dims(b, 0)

	# Only summing over the third dimension, while preserving the first two dimensions
	dist =	tf.reduce_sum(tf.squared_difference(a_expand, b_expand), axis=2)
	return dist

def kNN(dist, k):
	dist =		tf.negative(dist)
	val, ind =	tf.nn.top_k(tf.reduce_sum(dist, axis=1), k=K)
	respon =	tf.Variable(tf.zeros(10))
	respon =	tf.scatter_update(respon, ind, [1./k]*k)
	return respon

# Establish the main function to test and verify the code
if __name__ == '__main__':
	# Amount of data points for first input matrix
	B =	10
	# Data point for test input matrix
	C =	1
	# Amount of features per data point
	N =	3
	# Number of nearest neighors
	# make it multiple to test out the multiple indices selecting
	K =	3

	a =	tf.placeholder('float', [B, N]) # could use None but enforce row checking
	b =	tf.placeholder('float', [C, N]) # could use None but enforce row checking

	# Generate two inputs
	A = np.random.rand(B, N)
	B = np.random.rand(C, N)

	# Squared Euclidean Distance
	dist =	squared_dist(a, b)

	# List of responsibilities
	respon =	kNN(dist, K)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		dist_res, respon_res = sess.run([dist, respon], feed_dict={a: A, b: B})
		print("The resulting matrix (expecting dimension of BxC):\n{}".format(dist_res))
		print("\nThe resulting NN:\n {}".format(respon_res))
