# Wei Cui Leon Chen
# ECE521 A2 Question2.2
# Feb 24th 2017

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# hyper-parameters:
minibatch_size = 500
epoch_amount = 200

# Helper function for 2.2 part 1
def layer_wise_activation_computation(input, num_hidden_units):
	# get the input feature amounts
	# assume input tensor follows input_amount X feature_amount shape
	input_amount, feature_amount = input.get_shape().as_list()
	# initialize weights and biases
	with tf.variable_scope("layer_wise"):
		weights = tf.get_variable("weights", shape=[feature_amount, num_hidden_units], initializer=tf.contrib.layers.xavier_initializer())
		biases = tf.get_variable("biases", shape=[num_hidden_units], initializer=tf.constant_initializer(0.0))
		# perform weighted_sum
		z = tf.add( tf.matmul(tf.to_float(input), weights), biases)
	# return the populated weighted_sum for the hidden layer
	return z, weights

# Main network construction
class fully_connected_network:
	# global parameters
	output_classes = 10
	learning_rate = 0.001 # should try with 3 different values
	weight_decay_factor = 3e-4

	# initialization function
	def __init__(self, inputs, targets):
		# only input required is the input tensor
		self.inputs = inputs
		self.targets = targets
		self.hidden_units = 1000
		self.forward_prop()
		self.backward_prop()
	# Forward propagation
	def forward_prop(self, output_amount=output_classes):
		self.inputs_flatten = tf.reshape(self.inputs, [-1, 28*28])
		# input to hidden layer
		with tf.variable_scope("input_to_hidden"):
			hidden_z, weights = layer_wise_activation_computation(self.inputs_flatten, self.hidden_units)
			self.hidden = tf.nn.relu(hidden_z)
			self.hidden_weights = weights
		# From hidden layer to output
		with tf.variable_scope("hidden_to_output"):
			output_z, weights = layer_wise_activation_computation(self.hidden, output_amount)
			self.output = tf.nn.softmax(output_z)
			self.output_weights = weights

	# Backward propagation
	def backward_prop(self, l_rate = learning_rate, output_amount=output_classes):
		# compute the loss function
		self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=tf.one_hot(self.targets, depth=output_amount, axis=-1))
		self.weight_loss = (tf.nn.l2_loss(self.hidden_weights) + tf.nn.l2_loss(self.output_weights)) * self.weight_decay_factor
		self.total_loss = self.cross_entropy_loss + self.weight_loss
		# Compute classification accuracy
		self.correct_pred = tf.equal(tf.argmax(self.output, -1), tf.transpose(tf.to_int64(self.targets)))
		# define optimizer
		optimizer = tf.train.AdamOptimizer(l_rate)
		self.train_step = optimizer.minimize(self.total_loss)

if(__name__=='__main__'):
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


	# parse the data into minibatches
	# trainData_batches, trainTarget_batches = np.reshape(trainData, [-1, minibatch_size]), np.reshape(trainTarget, [-1, minibatch_size])
	# validData_batches, validTarget_batches = np.reshape(validData, [-1, minibatch_size]), np.reshape(validTarget, [-1, minibatch_size])
	# testData_batches, testTarget_batches = np.reshape(testData, [-1, minibatch_size]), np.reshape(testTarget, [-1, minibatch_size])
	trainData_batches = np.reshape(trainData, [-1, minibatch_size, 28, 28]); trainTarget_batches = np.reshape(trainTarget, [-1, minibatch_size, 1])
	train_batch_amount = np.shape(trainData_batches)[0]
	# convert all label files into column vectors
	trainTarget, validTarget, testTarget = np.reshape(trainTarget, [-1,1]), np.reshape(validTarget, [-1,1]), np.reshape(testTarget, [-1,1])

	# create an instance
	input_placeholder = tf.placeholder(tf.int32, [None, 28, 28], "input_placeholder")
	target_placeholder = tf.placeholder(tf.int32, [None, 1], "target_placeholder")
	fcn = fully_connected_network(input_placeholder, target_placeholder)

	# initialize the saver object for checkpoint files
	saver = tf.train.Saver(max_to_keep=4)
	# start a tensorflow session
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		cross_entropy_train_avg =	np.zeros(epoch_amount)
		cross_entropy_valid_avg =	np.zeros(epoch_amount)
		cross_entropy_test_avg =	np.zeros(epoch_amount)
		class_train_avg =	np.zeros(epoch_amount)
		class_valid_avg =	np.zeros(epoch_amount)
		class_test_avg =	np.zeros(epoch_amount)

		for i in range(epoch_amount):
			# train with training set
			for j in range(train_batch_amount):
				correct_pred, cross_entropy_loss, _ = sess.run([fcn.correct_pred, fcn.cross_entropy_loss, fcn.train_step], feed_dict={fcn.inputs:trainData_batches[j], fcn.targets:trainTarget_batches[j]})
				cross_entropy_train_avg[i] += np.average(cross_entropy_loss) / train_batch_amount
				class_train_avg[i] += np.average(correct_pred) / train_batch_amount

			# error with validation set
			correct_pred, cross_entropy_loss = sess.run([fcn.correct_pred, fcn.cross_entropy_loss], feed_dict={fcn.inputs: validData, fcn.targets: validTarget})
			cross_entropy_valid_avg[i] = np.average(cross_entropy_loss)
			class_valid_avg[i] = np.average(correct_pred)

			# Testing on test sets
			correct_pred, cross_entropy_loss = sess.run([fcn.correct_pred, fcn.cross_entropy_loss], feed_dict={fcn.inputs: testData, fcn.targets: testTarget})
			cross_entropy_test_avg[i] = np.average(cross_entropy_loss)
			class_test_avg[i] = np.average(correct_pred)

			if((i % 10)==0):
				print("At epoch {}, the avg cross entropy loss for each sample among training minibatches is: {}".format(i, cross_entropy_train_avg[i]))
				print("At epoch {}, the avg classification accuracy for each sample among training minibatches is: {}".format(i, class_train_avg[i]))
				print("At epoch {}, the avg cross entropy loss for each sample among validating set is: {}".format(i, cross_entropy_valid_avg[i]))
				print("At epoch {}, the avg classification accuracy for each sample among training minibatches is: {}".format(i, class_valid_avg[i]))
				print("At epoch {}, the avg cross entropy loss among testing minibatches is: {}".format(i, cross_entropy_test_avg[i]))
				print("At epoch {}, the avg classification accuracy among testing minibatches is: {}".format(i, class_test_avg[i]))

			if( ((float(i)/epoch_amount) % 0.25) == 0 and i != 0):
				print("At epoch {}, arrived at a checkpoint. Saving the model...".format(i))
				checkpoint_index = int((float(i)/epoch_amount) / 0.25)
				# save the model to checkpoint file
				save_path = saver.save(sess, './saved_models/fc_network_model', global_step=checkpoint_index)
				print("Model saved at {} for {}percent of training progress".format(save_path, checkpoint_index))

		# Plot the data
		plt.figure(1)
		plt.plot(cross_entropy_train_avg, label='Training')
		plt.plot(cross_entropy_valid_avg, label='Validation')
		plt.plot(cross_entropy_test_avg, label='Testing')
		plt.scatter(100, cross_entropy_train_avg[99], label='Training Early')
		plt.scatter(100, cross_entropy_valid_avg[99], label='Validation Early')
		plt.scatter(100, cross_entropy_test_avg[99], label='Testing Early')
		plt.xlabel('Epoch')
		plt.ylabel('Cross Entropy')
		plt.title('Cross Entropy for Learning Rate of 0.001')
		plt.legend(loc='upper right')

		plt.figure(2)
		plt.plot(class_train_avg, label='Training')
		plt.plot(class_valid_avg, label='Validation')
		plt.plot(class_test_avg, label='Testing')
		plt.scatter(150, class_train_avg[149], label='Training Early')
		plt.scatter(150, class_valid_avg[149], label='Validation Early')
		plt.scatter(150, class_test_avg[149], label='Testing Early')
		plt.xlabel('Epoch')
		plt.ylabel('Classification %')
		plt.title('Classification Accuracy for Learning Rate of 0.001')
		plt.legend(loc='lower right')

		print('Training accuracy at early stopping:	', class_train_avg[149])
		print('Validation accuracy at early stopping:	', class_valid_avg[149])
		print('Testing accuracy at early stopping:	', class_test_avg[149])

	plt.show()
