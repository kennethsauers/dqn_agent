import tensorflow as tf
import numpy as np
import os

class brain():
	def __init__(self, name = 'brain'):
		self.name = name
		self.num_out = 10
		self.lr = .003
		self.W = 400
		self.H = 120
		self.n_channels = 4
		tf.reset_default_graph()

		self.make_dir()
		self.build_model()
		self.sess = tf.Session()
		self.saver = tf.train.Saver()
	def build_model(self):

		weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.003)
		activation  = tf.nn.relu
		padding     = 'SAME'
		conv2d = tf.layers.conv2d
		dense = tf.layers.dense
		flatten = tf.contrib.layers.flatten
		self.input_layer = tf.placeholder(shape=[None, self.W, self.H, self.n_channels], dtype=tf.float32, name="input_layer")

		# 1st conv layer
		NN = conv2d(name="conv1", inputs=self.input_layer,
		        filters=16, kernel_size=3, strides=2,
		        padding=padding, kernel_initializer=weight_init,
		        activation=activation)
		# 2nd conv layer
		NN = conv2d(name="conv2", inputs=NN,
		        filters=32, kernel_size=3, strides=2,
		        padding=padding, kernel_initializer=weight_init,
		        activation=activation)

		# 3rd conv layer
		NN = conv2d(name="conv3", inputs=NN,
		        filters=64, kernel_size=3, strides=1,
		        padding=padding, kernel_initializer=weight_init,
		        activation=activation)

		# Flatten the output of the 3rd convolutional layer to input into fc layer
		NN = flatten(NN)

		# 1st fully connected layer
		NN = dense( name='fc1', inputs=NN, units=1024,
		        kernel_initializer=weight_init,
		        activation=activation)
		# 2nd fc layer
		NN = dense( name='fc2', inputs=NN, units=1024,
		        kernel_initializer=weight_init,
		        activation=activation)
		NN = dense( name='fc3', inputs=NN, units=1024,
		        kernel_initializer=weight_init,
		        activation=activation)
		# Output layer
		NN = dense( name='fc_out', inputs=NN, units=self.num_out,
		        kernel_initializer=weight_init,
		        activation=None)

		self.q_values = NN
		self.prediction = tf.argmax(self.q_values, 1)

		self.q_target = tf.placeholder(shape=[None],dtype=tf.float32)
		self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
		self.actions_onehot = tf.one_hot(self.actions,self.num_out,dtype=tf.float32)

		self.q_predicted = tf.reduce_sum(tf.multiply(self.q_values, self.actions_onehot), axis=1)

		self.td_error = tf.square(self.q_target - self.q_predicted)
		self.loss = tf.reduce_mean(self.td_error)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

		return True

	def make_dir(self):
		self.model_path = 'model/' + self.name
		self.data_path = 'data/' + self.name
		if not os.path.exists(self.model_path):
			os.makedirs(self.model_path)

		if not os.path.exists(self.data_path):
			os.makedirs(self.data_path)
		return True
	def Q(self, state):
		return self.sess.run(self.q_values, feed_dict = {self.input_layer: state})

	def V(self, state):
		q_values = self.Q(state)
		v_values = [max(q_values[i]) for i in range(len(q_values))]
		return v_values

	def predict(self, state):
		q_values = self.Q(state)
		actions = [np.argmax(q_values[i]) for i in range(len(state))]
		return actions

	def load(self):
		self.saver.restore(self.sess, save_path = self.model_path)	

	def save(self):
		self.saver.save(self.sess, save_path = self.model_path)

	def close_session(self):
		self.sess.close()

	def optimize_step(self, batch):
		batch_size = len(batch)
		states      = [batch[i][0] for i in range(batch_size)]
		actions     = [batch[i][1] for i in range(batch_size)]
		rewards     = [batch[i][2] for i in range(batch_size)]
		next_states = [batch[i][3] for i in range(batch_size)]
		terminal    = [batch[i][4] for i in range(batch_size)]

		V = self.V(next_states)

		q_target = [rewards[i] + (0 if terminal[i] else self.GAMMA*V[i]) \
		            for i in range(batch_size)]

		feed_dict = {self.input_layer: states, self.q_target: q_target, self.actions: actions}

		_, loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
		return loss