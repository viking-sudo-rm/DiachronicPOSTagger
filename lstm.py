from __future__ import division

import tensorflow as tf
import numpy as np
import os, sys
from tensorflow.contrib import rnn

# To stack more LSTM layers, just add more sizes to this list
LAYERS = [512]

# Maximum length of sentences
MAX_LEN = 50

# Size of input embedding vectors
EMBED_DIM = 300

# Number of parts of speech
N_POS = 423

# Hyperparameters
LR = 0.01
N_EPOCHS = 30
BATCH_SIZE = 10

class Dataset:

	""" Wrapper class for storing data. """

	def __init__(self, X_word, X_year, Y_pred):
		self.X_word = X_word
		self.X_year = X_year
		self.Y_pred = Y_pred

	@staticmethod
	def load(self, path):

		X_word_path = os.path.join(path, "X_word")
		X_word_path = os.path.join(path, "X_year")
		X_word_path = os.path.join(path, "Y")

		X_word_arrays = []
		X_year_arrays = []
		Y_arrays = []

		for dirpath, dirnames, filenames in os.walk(X_word_path):
			for filename in filenames:
				print("WORD" + filename)
				X_word_array = np.load(os.path.join(X_word_path, filename))
				X_word_arrays.append(X_word_array)

		for dirpath, dirnames, filenames in os.walk(X_year_path):
			for filename in filenames:
				print("YEAR" + filename)
				X_year_array = np.load(os.path.join(X_year_path, filename))
				X_year_arrays.append(X_year_array)

		for dirpath, dirnames, filenames in os.walk(Y_path):
			for filename in filenames:
				print("PATH" + filename)
				Y_array = np.load(os.path.join(Y_path, filename))
				Y_arrays.append(Y_array)

		return Dataset(
			np.concatenate(X_word_arrays, axis=0),
			np.concatenate(X_year_arrays, axis=0),
			np.concatenate(Y_arrays, axis=0),
		)

	def shuffle(self):
		permutation = np.random.permutation(self.X_word.shape[0])
		self.X_word = self.X_word[permutation, :, :]
		self.X_year = self.X_year[permutation, :]
		self.Y_pred = self.Y_pred[permutation, :]

	def iter_batches(self):
		for i in xrange(0, len(self.X_word) - BATCH_SIZE, BATCH_SIZE):
			yield (
				self.X_word[i:i+BATCH_SIZE, :, :],
				self.X_year[i:i+BATCH_SIZE, :],
				self.Y_pred[i:i+BATCH_SIZE, :],
			)

	def get_n_batches(self):
		return len(self.X_word) // BATCH_SIZE


class TemporalLanguageModel:

	def add_graph(self):

		self.X_word = tf.placeholder(tf.float32, [None, MAX_LEN, EMBED_DIM])
		self.X_year = tf.placeholder(tf.float32, [None, MAX_LEN])
		self.Y_pred = tf.placeholder(tf.int32, [None, MAX_LEN])

		X = tf.concat([self.X_word, tf.expand_dims(self.X_year, 2)], axis=2)

		rnn_layers = [rnn.LSTMCell(size) for size in LAYERS]
		multi_rnn_cell = rnn.MultiRNNCell(rnn_layers)

		H, _ = tf.nn.dynamic_rnn(
			cell=multi_rnn_cell,
			inputs=X, 
			dtype=tf.float32
		)

		self.Y = tf.contrib.layers.fully_connected(
			inputs=H,
			num_outputs=N_POS,
		)

		self.acc = tf.reduce_mean(tf.nn.top_k(self.Y_pred, k=1))

		log_p = tf.gather(tf.log(self.Y), self.Y_pred)
		self.log_perp = -tf.reduce_mean(log_p)

		self.loss = tf.losses.sparse_softmax_cross_entropy(
			labels=self.Y_pred,
			logits=self.Y,
		)

		self.train_step = tf.train.AdamOptimizer(LR).minimize(self.loss)

	def train(self, session, train_data, dev_data):
		session.run(tf.global_variables_initializer())
		n_batches = train_data.get_n_batches()
		for i in xrange(N_EPOCHS):
			train_data.shuffle()
			loss = 0.
			for batch_X_word, batch_X_year, batch_Y_pred in train_data.iter_batches():
				d_loss, _ = session.run([self.loss, self.train_step], feed_dict={
					self.X_word: batch_X_word,
					self.X_year: batch_X_year,
					self.Y_pred: batch_Y_pred,
				})
				loss += d_loss
			print "#{} TRAIN : loss={:.3f}".format(i, loss / n_batches)
			dev_loss, dev_acc, dev_log_perp = session.run([self.loss, self.acc, self.log_perp], feed_dict={
				self.X_word: dev_data.X_word,
				self.X_year: dev_data.X_year,
				self.Y_pred: dev_data.Y_pred,
			})
			print "#{} DEV : loss={:.3f} : acc={:.3f} : log_perp={:.3f}".format(i, dev_loss, dev_acc, dev_log_perp)

def main():

	# train_data = Dataset.load(TRAIN_PATH)
	# dev_data = Dataset.load(DEV_PATH)
	# test_data = Dataset.load(TEST_PATH)

	train_data = Dataset(
		np.random.uniform(size=[100, MAX_LEN, EMBED_DIM]),
		np.random.uniform(size=[100, MAX_LEN]),
		np.random.uniform(size=[100, MAX_LEN], high=100),
	)

	dev_data = Dataset(
		np.random.uniform(size=[100, MAX_LEN, EMBED_DIM]),
		np.random.uniform(size=[100, MAX_LEN]),
		np.random.uniform(size=[100, MAX_LEN], high=100),
	)

	session = tf.Session()

	model = TemporalLanguageModel()
	model.add_graph()
	model.train(session, train_data, dev_data)


if __name__ == "__main__":
	main()