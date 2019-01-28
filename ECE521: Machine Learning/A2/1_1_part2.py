# Wei Cui Leon Chen
# ECE521 A2 Question1.2
# Feb 24th 2017

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# hyper-parameters
weight_decay_factor = 0.01
batch_size = 500
learning_rate = 0.005
epoch_amount = 100


## Tensorflow computational graph

# placeholders
x = tf.placeholder(tf.int32, [None, 28*28], "input_placeholder")
y = tf.placeholder(tf.int32, [None, 1], "target_placeholder")

# declare variables & computing z
with tf.variable_scope("logistic_regression"):
	weights = tf.get_variable("weights", [28*28, 1], initializer=tf.contrib.layers.xavier_initializer())
	bias = tf.get_variable("bias", [1], initializer=tf.constant_initializer(0.0))
	# bias would be broadcasted to be added on each sample within minibatch
	z = tf.add(tf.matmul(tf.to_float(x), weights), bias)

# compute cross entropy loss
cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=tf.to_float(y))
# weight decay loss
weight_decay_loss = tf.nn.l2_loss(weights) * weight_decay_factor
total_loss = cross_entropy_loss + weight_decay_loss
# define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
# define train step
train_step = optimizer.minimize(cross_entropy_loss)


## Testing and training

if(__name__ == "__main__"):
	# prepare input data
	with np.load("notMNIST.npz") as data:
		Data, Target = data["images"], data["labels"]
		np.random.seed(521)
		randIndx = np.arange(len(Data))
		np.random.shuffle(randIndx)
		Data = Data[randIndx]/255.
		Target = Target[randIndx]
		trainData, trainTarget = Data[:15000], Target[:15000]
		validData, validTarget = Data[15000:16000], Target[15000:16000]
		testData, testTarget = Data[16000:], Target[16000:]

	# Flatten images into 1D vector
	trainData =	np.reshape(trainData, [trainData.shape[0], -1])
	validData =	np.reshape(validData, [validData.shape[0], -1])
	testData =	np.reshape(testData, [testData.shape[0], -1])

	# Resize to 2D array
	trainTarget =	np.reshape(trainTarget, [trainTarget.shape[0], 1])
	validTarget =	np.reshape(validTarget, [validTarget.shape[0], 1])
	testTarget =	np.reshape(testTarget, [testTarget.shape[0], 1])

	# parse the data into minibatches
	trainData_batches =		np.reshape(trainData, [-1, batch_size, 784])
	trainTarget_batches =	np.reshape(trainTarget, [-1, batch_size, 1])
	batch_amount =			trainData_batches.shape[0]

	#if validData.shape[0] % batch_amount != 0:
	#	validData =		np.pad(validData, ((0, batch_amount - validData.shape[0] % batch_amount), (0, 0)), 'wrap')
	#	validTarget =	np.pad(validTarget, ((0, batch_amount - validTarget.shape[0] % batch_amount), (0, 0)), 'wrap')
	#validData_batches =		np.reshape(validData, [batch_amount, -1, 784])
	#validTarget_batches =	np.reshape(validTarget, [batch_amount, -1, 1])

	# testData_batches =	np.reshape(testData, [-1, batch_size, 784])
	# testTarget_batches =	np.reshape(testTarget, [-1, batch_size, 1])
	# test_batch_amount =	testData_batches.shape[0]

	# start a tensorflow session
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		cross_entropy_train_avg =	np.zeros(epoch_amount)
		cross_entropy_valid_avg =	np.zeros(epoch_amount)
		cross_entropy_test_avg =	np.zeros(epoch_amount)

		for epoch in range(epoch_amount):
			for batch in range(batch_amount):
				cross_entropy_train, _, waits =	sess.run([cross_entropy_loss, train_step, weights], feed_dict={x:trainData_batches[batch, :], y:trainTarget_batches[batch, :]})
				cross_entropy_train_avg[epoch] +=	cross_entropy_train.sum() / (trainData.shape[0])

			cross_entropy_valid =		sess.run(cross_entropy_loss, feed_dict={x: validData, y: validTarget})
			cross_entropy_valid_avg[epoch] +=	cross_entropy_valid.sum() / (validData.shape[0])
			cross_entropy_test =		sess.run(cross_entropy_loss, feed_dict={x: testData, y: testTarget})
			cross_entropy_test_avg[epoch] +=	cross_entropy_test.sum() / (testData.shape[0])

			#if ((epoch+1) % 10 == 0):
			#	print("At epoch {}, the average cross entropy loss among minibatches is:	{}".format(epoch+1, cross_entropy_train_avg[epoch]))
			#	print("At epoch {}, the validation cross entropy loss among minibatches is:	{}".format(epoch+1, cross_entropy_valid_avg[epoch]))
			#	print("At epoch {}, the testing cross entropy loss among minibatches is:	{}\n".format(epoch+1, cross_entropy_test_avg[epoch]))

		plt.plot(cross_entropy_train_avg, label='Training')
		plt.plot(cross_entropy_valid_avg, label='Validation')
		plt.plot(cross_entropy_test_avg, label='Testing')
		print(waits)
	plt.legend(loc='upper right')
	plt.show()
