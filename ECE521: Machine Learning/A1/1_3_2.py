# Question 1.3 Part 1
import matplotlib.pyplot as plt
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
	# Want nearest k, so take negative of squared distance and take largest k neighours
	dist =		tf.negative(dist)
	val, ind =	tf.nn.top_k(tf.reduce_sum(dist, axis=1), k=k)

	# Update the responsibilities
	single =	tf.to_float(tf.truediv(1, k))	# Responsibility of on each neighour
	respon =	tf.Variable(tf.zeros(80))
	respon =	tf.scatter_update(respon, ind, tf.fill(tf.expand_dims(k, 0), single))
	return respon

# Establish the main function to test and verify the code
if __name__ == '__main__':
	# Make data1D
	np.random.seed(521)
	Data =		np.linspace(1.0 , 10.0 , num =100) [:, np.newaxis]
	Target =	np.sin(Data) + 0.1 * np.power(Data, 2) + 0.5 * np.random.randn(100, 1)
	randIdx =	np.arange(100)
	np.random.shuffle(randIdx)
	trainData, trainTarget =	Data[randIdx[:80]], Target[randIdx[:80]]
	validData, validTarget =	Data[randIdx[80:90]], Target[randIdx[80:90]]
	testData, testTarget =		Data[randIdx[90:100]], Target[randIdx[90:100]]

	# Plotting Data
	X =	np.linspace(0.0, 11.0, num = 110)[:, np.newaxis]
	Y =	np.zeros(110)

	# Amount of data points for first input matrix
	M =	100

	# K values to try
	trial_k = [1, 3, 5, 50]

	a = tf.placeholder(tf.float32, [None, 1])	# First input
	b = tf.placeholder(tf.float32, [1, 1])		# Second input
	t = tf.placeholder(tf.float32, [None, 1])	# Target
	k =	tf.placeholder(tf.int32, ())			# Hyper-parameter

	# Squared Euclidean Distance
	dist =	squared_dist(a, b)

	# List of responsibilities
	respon =	kNN(dist, k)

	# Prediction
	pred =	tf.matmul(tf.transpose(t), tf.expand_dims(respon, 1))

	for K in trial_k:
		# MSE Errors restarts after each K change
		trainMSE =	0;
		validMSE =	0;
		testMSE =	0;

		with tf.Session() as sess:
			# Training
			for m in range(trainData.size):
				sess.run(tf.global_variables_initializer())
				pred_res =	sess.run(pred, feed_dict={a: trainData, b: trainData[m].reshape(1, 1), t: trainTarget, k: K})
				trainMSE =	trainMSE + np.linalg.norm(pred_res - trainTarget[m])**2/(2*trainData.size)
			# Validation
			for m in range(validData.size):
				sess.run(tf.global_variables_initializer())
				pred_res =	sess.run(pred, feed_dict={a: trainData, b: validData[m].reshape(1, 1), t: trainTarget, k: K})
				validMSE =	validMSE + np.linalg.norm(pred_res - validTarget[m])**2/(2*validData.size)
			# Testing
			for m in range(testData.size):
				sess.run(tf.global_variables_initializer())
				pred_res =	sess.run(pred, feed_dict={a: trainData, b: testData[m].reshape(1, 1), t: trainTarget, k: K})
				testMSE =	testMSE + np.linalg.norm(pred_res - testTarget[m])**2/(2*testData.size)
			# Estimate plot data
			for m in range(X.size):
				sess.run(tf.global_variables_initializer())
				pred_res =	sess.run(pred, feed_dict={a: trainData, b: X[m].reshape(1, 1), t: trainTarget, k: K})
				Y[m] = 		np.linalg.norm(pred_res)
			plt.plot(X, Y, label='K='+str(K))	# Plotting

		# Output performance
		print('K={}, Training MSE: {}'.format(K, trainMSE))
		print('K={}, Validation MSE: {}'.format(K, validMSE))
		print('K={}, Testing MSE: {}\n'.format(K, testMSE))


	# Show plots
	plt.xlabel('Input')
	plt.ylabel('Output')
	plt.title('Comparsion of K Values')
	plt.legend(loc='upper left')
	plt.show()
