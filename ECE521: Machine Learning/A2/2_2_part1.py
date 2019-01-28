# Wei Cui Leon Chen
# ECE521 A2 Question2.2
# Feb 24th 2017

import numpy as np
import tensorflow as tf

# Helper function for 2.2 part 1
def layer_wise_activation_computation(input, num_hidden_units):
	# get the input feature amounts
	# assume input tensor follows input_amount X feature_amount shape
	num_input_feature = tf.shape(input)[1]
	# initialize weights and biases
	with tf.variable_scope("layer_wise"):
		weights = tf.get_variable("weights", shape=[num_input_feature, num_hidden_units], initializer=tf.contrib.layers.xavier_initializer())
		biases = tf.get_variable("biases", shape=[num_hidden_units], initializer=tf.constant_initializer(0.0))
		# perform weighted_sum
		z = tf.add( tf.matmul(input, weights), biases)
	# return the populated weighted_sum for the hidden layer
	return z
