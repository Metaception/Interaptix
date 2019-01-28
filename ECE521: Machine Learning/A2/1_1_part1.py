# Wei Cui Leon Chen
# ECE521 A2 Question1.1
# Feb 23rd 2017

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# hyper-parameters
weight_decay_factor = 0.01
batch_size = 500
learning_rate = 0.0001
epoch_amount = 1000


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
cross_entropy_loss =	tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=tf.to_float(y))
# Compute classification accuracy
correct_pred =			tf.equal(tf.round(tf.nn.sigmoid(z)), tf.to_float(y))
# weight decay loss
weight_decay_loss = tf.nn.l2_loss(weights) * weight_decay_factor
total_loss = cross_entropy_loss + weight_decay_loss
# define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# define train step
train_step = optimizer.minimize(total_loss)


## Testing and training

if(__name__ == "__main__"):
	# prepare input data
	with np.load("notMNIST.npz") as data :
		Data, Target = data["images"], data["labels"]
		posClass = 2
		negClass = 9
		dataIndx = (Target==posClass) + (Target==negClass)
		Data = Data[dataIndx]/255.
		Target = Target[dataIndx].reshape(-1, 1)
		Target[Target==posClass] = 1
		Target[Target==negClass] = 0
		np.random.seed(521)
		randIndx = np.arange(len(Data))
		np.random.shuffle(randIndx)
		Data, Target = Data[randIndx], Target[randIndx]
		trainData, trainTarget = Data[:3500], Target[:3500]
		validData, validTarget = Data[3500:3600], Target[3500:3600]
		testData, testTarget = Data[3600:], Target[3600:]
	# Flatten images into 1D vector
	trainData =	np.reshape(trainData, [trainData.shape[0], -1])
	validData =	np.reshape(validData, [validData.shape[0], -1])
	testData =	np.reshape(testData, [testData.shape[0], -1])

	# parse the data into minibatches
	trainData_batches =		np.reshape(trainData, [-1, batch_size, 784])
	trainTarget_batches =	np.reshape(trainTarget, [-1, batch_size, 1])
	batch_amount =			trainData_batches.shape[0]

	# start a tensorflow session
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		cross_entropy_train_avg =	np.zeros(epoch_amount)
		cross_entropy_valid_avg =	np.zeros(epoch_amount)
		cross_entropy_test_avg =	np.zeros(epoch_amount)
		class_train_avg =	np.zeros(epoch_amount)
		class_valid_avg =	np.zeros(epoch_amount)
		class_test_avg =	np.zeros(epoch_amount)

		for epoch in range(epoch_amount):
			# Training
			for batch in range(batch_amount):
				class_train, cross_entropy_train, _ =	sess.run([correct_pred, cross_entropy_loss, train_step], feed_dict={x:trainData_batches[batch, :], y:trainTarget_batches[batch, :]})
				cross_entropy_train_avg[epoch] +=		cross_entropy_train.sum() / (trainData.shape[0])
				class_train_avg[epoch] +=				(class_train.sum() / class_train.shape[0]) / batch_amount

			# Validation
			class_valid, cross_entropy_valid =	sess.run([correct_pred, cross_entropy_loss], feed_dict={x: validData, y: validTarget})
			cross_entropy_valid_avg[epoch] =	cross_entropy_valid.sum() / (validData.shape[0])
			class_valid_avg[epoch] =			class_valid.sum() / class_valid.shape[0]

			# Testing
			class_test, cross_entropy_test =	sess.run([correct_pred, cross_entropy_loss], feed_dict={x: testData, y: testTarget})
			cross_entropy_test_avg[epoch] =		cross_entropy_test.sum() / (testData.shape[0])
			class_test_avg[epoch] =				class_test.sum() / class_test.shape[0]

			'''
			if ((epoch+1) % 10 == 0):
				print("\nAt epoch {}".format(epoch+1))
				print("Mean training cross entropy loss is:		{}".format(cross_entropy_train_avg[epoch]))
				print("Validation cross entropy loss is:		{}".format(cross_entropy_valid_avg[epoch]))
				print("Testing cross entropy loss is:			{}".format(cross_entropy_test_avg[epoch]))
				print("Mean training classification accuracy is:	{}".format(class_train_avg[epoch]))
				print("Validation classification accuracy is:		{}".format(class_valid_avg[epoch]))
				print("Testing classification accuracy is:		{}".format(class_test_avg[epoch]))
			'''
		# Plot the data
		plt.figure(1)
		plt.plot(cross_entropy_train_avg, label='Training')
		plt.plot(cross_entropy_valid_avg, label='Validation')
		plt.plot(cross_entropy_test_avg, label='Testing')
		plt.xlabel('Cross Entropy')
		plt.ylabel('Epoch')
		plt.title('Cross Entropy for SGD')
		plt.legend(loc='upper right')

		plt.figure(2)
		plt.plot(class_train_avg, label='Training')
		plt.plot(class_valid_avg, label='Validation')
		plt.plot(class_test_avg, label='Testing')
		plt.xlabel('Classification %')
		plt.ylabel('Epoch')
		plt.title('Classification Accuracy for SGD')
		plt.legend(loc='upper right')

		# Get best classification accuracy
		print("Best classification accuracy of {} at epoch {}:".format(class_test_avg.max(), class_test_avg.argmax()))

	plt.show()
