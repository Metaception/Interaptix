# Wei Cui Leon Chen
# ECE521 A2 Question1.1 Part 3
# Feb 23rd 2017

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# hyper-parameters
weight_decay = 0 # set weight_decay to zero as specified by the question
batch_size = 500
learning_rate = 0.0001
epoch_amount = 1000


## Tensorflow computational graph

# placeholders
x = tf.placeholder(tf.int32, [None, 28*28], "input_placeholder")
y = tf.placeholder(tf.int32, [None, 1], "target_placeholder")

# declare variables & computing z
with tf.variable_scope("least_square_regression"):
	weights = tf.get_variable("weights", [28*28, 1], initializer=tf.contrib.layers.xavier_initializer())
	bias = tf.get_variable("bias", [1], initializer=tf.constant_initializer(0.0))
	# bias would be broadcasted to be added on each sample within minibatch
	z = tf.add(tf.matmul(tf.to_float(x), weights), bias)

# With least square regression, use L2 loss
l2_loss = tf.nn.l2_loss(tf.subtract(z, tf.to_float(y)))
indiv_loss = tf.square(z - tf.to_float(y))
# Predictions
prediction = z
# Compute classification accuracy
correct_pred = tf.equal(tf.nn.relu(tf.sign(z -0.5)), tf.to_float(y))
# define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
# define train step
train_step = optimizer.minimize(l2_loss)


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
		l2_train_avg =	np.zeros(epoch_amount)
		l2_valid_avg =	np.zeros(epoch_amount)
		l2_test_avg =	np.zeros(epoch_amount)
		class_train_avg =	np.zeros(epoch_amount)
		class_valid_avg =	np.zeros(epoch_amount)
		class_test_avg =	np.zeros(epoch_amount)

		for epoch in range(epoch_amount):
			# Training
			for batch in range(batch_amount):
				class_train, l2_train, _ =	sess.run([correct_pred, l2_loss, train_step], feed_dict={x:trainData_batches[batch, :], y:trainTarget_batches[batch, :]})
				l2_train_avg[epoch] +=		np.average(l2_train) / trainData.shape[0]
				class_train_avg[epoch] +=	np.average(class_train) / batch_amount

			# Validation
			class_valid, l2_valid =		sess.run([correct_pred, l2_loss], feed_dict={x: validData, y: validTarget})
			l2_valid_avg[epoch] =		np.average(l2_valid) / validData.shape[0]
			class_valid_avg[epoch] =	np.average(class_valid)

			# Testing
			pred, class_test, losses, l2_test =	sess.run([prediction, correct_pred, indiv_loss, l2_loss], feed_dict={x: testData, y: testTarget})
			l2_test_avg[epoch] =	np.average(l2_test) / testData.shape[0]
			class_test_avg[epoch] =	np.average(class_test)

			# Plot the data
			plt.figure(3)
			plt.scatter(pred, losses, s=1, marker='.')

		# Plot the data
		plt.figure(1)
		plt.plot(l2_train_avg, label='Training')
		plt.plot(l2_valid_avg, label='Validation')
		plt.plot(l2_test_avg, label='Testing')
		plt.xlabel('Epoch')
		plt.ylabel('L2 Loss')
		plt.title('L2 Loss without Weight Decay')
		plt.legend(loc='upper right')

		plt.figure(2)
		plt.plot(class_train_avg, label='Training')
		plt.plot(class_valid_avg, label='Validation')
		plt.plot(class_test_avg, label='Testing')
		plt.xlabel('Epoch')
		plt.ylabel('Classification %')
		plt.title('Classification Accuracy without Weight Decay')
		plt.legend(loc='upper right')

		plt.figure(3)
		plt.xlabel('Prediction')
		plt.xlim([0, 1])
		plt.ylabel('L2 Loss')
		plt.ylim([0, 1])
		plt.title('L2 Losses')
		plt.legend(loc='upper right')
		plt.plot((0, 1), (0, 0), 'blue')

		# Get best classification accuracy
		best_class =	class_test_avg.argmax()
		print("Best classification accuracy of {} at epoch {}:".format(class_test_avg[best_class], best_class))

	plt.show()
