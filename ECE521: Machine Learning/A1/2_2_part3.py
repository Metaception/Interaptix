# This program is for question 2.2 of ECE521 A1
import numpy as np
import tensorflow as tf

# hyper-parameters
stddev_para = 0.01
# Selecting Adam optimizer, can start with a relatively high learning rate,
# as the Adam optimizer would adjust the learning rate
# throughout the training progress
learning_rate = 0.01
# mini-batch size
mini_batch_size = 50
# total training epoches
epoches = 500
# weight decay factor
weight_decay_const = 1

# Build the graph
# Declare placeholders
x = tf.placeholder(tf.float32, shape=(mini_batch_size, 64)) # int32 might work
y = tf.placeholder(tf.float32, shape=(mini_batch_size, 1)) # only binary output

# first of all, flatten the input into vectors for fully connected layer regression
# The input is already flatten
#x_flatten = tf.reshape(x, shape=[50, -1])


# perform linear regression
with tf.variable_scope("linear_regression"):
	weights = tf.get_variable("weights", [8*8*1, 1], initializer = tf.truncated_normal_initializer(stddev = stddev_para))
	biases = tf.get_variable("biases", [1], initializer=tf.constant_initializer(0))

	# for binary classification, using sigmoid for class scores
	# actually, just compute z here, as the activations would be computed with computing the loss function
	outputs = tf.add(tf.matmul(x, weights), biases)

# the output should be with dimension 50 X 1
#tf.assert_equal(tf.shape(outputs), [[50, 1]], data=outputs, summarize=50, message="regression output has abnormal dimension")
#outputs = tf.Print(outputs, [tf.shape(outputs)], message="printing outputs shape")

# as this is a binary classification, use cross_entropy
MSE_loss = tf.div(tf.nn.l2_loss(tf.sub(outputs, y)),  mini_batch_size)
# compute the weight decay loss
weight_decay = tf.mul( tf.reduce_sum(tf.square(weights)), weight_decay_const/2 )
# total loss function
total_loss = MSE_loss + weight_decay
# Use Tensorboard to visualize the training curve
tf.summary.scalar('total_loss', total_loss)
# optimizer op
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# define the optimize step
train_step = optimizer.minimize(total_loss)

if __name__ == '__main__':

	# load the training, validation, and testing data
	with np.load("tinymnist.npz") as data:
		trainData, trainTarget = data["x"], data["y"]
		validData, validTarget = data["x_valid"], data["y_valid"]
		testData, testTarget = data["x_test"], data["y_test"]

	# prepare the training data into minibatches of specified minibatch size
	amount_train = np.shape(trainData)[0]
        #print(np.shape(trainTarget))
	# drop the extra training samples if such exist that don't fit within minibatches
	amount_batch = int(np.floor(amount_train / mini_batch_size))
	trainData_parsed = []
	trainTarget_parsed = []
	for i in range(amount_batch):
		trainData_parsed.append(trainData[i*mini_batch_size: (i+1)*mini_batch_size])
		trainTarget_parsed.append(trainTarget[i*mini_batch_size: (i+1)*mini_batch_size])
		
	# prepare the validation data into minibatches of specified minibatch size
	amount_valid = np.shape(validData)[0]
	# drop the extra training samples if such exist that don't fit within minibatches
	amount_batch_valid = int(np.floor(amount_valid / mini_batch_size))
	validData_parsed = []
	validTarget_parsed = []
	for i in range(amount_batch_valid):
	        validData_parsed.append(validData[i*mini_batch_size: (i+1)*mini_batch_size])
		validTarget_parsed.append(validTarget[i*mini_batch_size: (i+1)*mini_batch_size])
		
	# prepare the testing data into minibatches of 50
	amount_test = np.shape(testData)[0]
	# drop the extra training samples if such exist that don't fit within minibatches
	amount_batch_test = int(np.floor(amount_test / mini_batch_size))
	testData_parsed = []
	testTarget_parsed = []
	for i in range(amount_batch_test):
		testData_parsed.append(testData[i*mini_batch_size: (i+1)*mini_batch_size])
		testTarget_parsed.append(testTarget[i*mini_batch_size: (i+1)*mini_batch_size])


	# start the session to run through the neural network
	with tf.Session() as sess:
		# merge all summary ops
		merged = tf.summary.merge_all()
		# define the summary writer for training curve
		train_writer = tf.train.SummaryWriter('logs', sess.graph)
		# start with initialize all variables
		sess.run(tf.global_variables_initializer())
		print("starts training the network")
		for i in range(epoches):
			avg_cross_entropy_loss = 0
			for j in range(amount_batch):
				cross_entropy_loss, _, summary = sess.run([total_loss, train_step, merged], feed_dict={x: trainData_parsed[j], y:trainTarget_parsed[j]})
				avg_cross_entropy_loss += cross_entropy_loss / amount_batch
			print("<-----Training Epoch #{}: average total loss between minibatches is: {}".format(i+1, avg_cross_entropy_loss))
                        # adding the average loss into the train writer for visualization
			train_writer.add_summary(summary, i)
		# finish training the network
		print("Finish training the network")
		
		# Start on validation set accuracy computation
		print("Start evaluating validation set accuracy")
		avg_accur_valid = 0
		for i in range(amount_batch_valid):
			# get the result for each minibatch
			classification_result = sess.run([outputs], feed_dict={x: validData_parsed[i], y: validTarget_parsed[i]})
			# make it binary
			classification_result_binary = []
			for single_result in classification_result:
			    if(single_result >= 0.5):
				classification_result_binary.append(1)
			    else:
				classification_result_binary.append(0)
			# convert the binary results into a column
			classification = np.reshape(np.array(classification_result_binary), shape=[-1, 1])
			
			# compute the accuracy rate:
			accur_rate = float(np.sum( np.abs(classification - np.array(validTarget_parsed)) )) / minibatch_size
			avg_accur_valid += accur_rate / minibatch_size
	        print("Validation accuracy average across all minibatches: {}".format(avg_accur_valid))
		
		# Start on testing set accuracy computation
		print("Start evaluating validation set accuracy")
		avg_accur_test = 0
		for i in range(amount_batch_test):
			# get the result for each minibatch
			classification_result = sess.run([outputs], feed_dict={x: testData_parsed[i], y: testTarget_parsed[i]})
			# make it binary
			classification_result_binary = []
			for single_result in classification_result:
			    if(single_result >= 0.5):
				classification_result_binary.append(1)
			    else:
				classification_result_binary.append(0)
			# convert the binary results into a column
			classification = np.reshape(np.array(classification_result_binary), shape=[-1, 1])
			
			# compute the accuracy rate:
			accur_rate = float(np.sum( np.abs(classification - np.array(validTarget_parsed)) )) / minibatch_size
			avg_accur_test += accur_rate / minibatch_size
	        print("Testing accuracy average across all minibatches: {}".format(avg_accur_test))
	# finish the script
	print("Script finished completely!")
