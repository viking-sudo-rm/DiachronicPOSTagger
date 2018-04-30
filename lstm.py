import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.utils import shuffle

# To stack more LSTM layers, just add more sizes to this list
LAYERS = [512]

# Maximum length of sentences
MAX_LEN = 100

# Size of input embedding vectors
EMBED_DIM = 300

# Number of words in vocabulary
VOCAB_DIM = 1024

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

	def shuffle(self):
		self.X_word, self.X_year, self.Y_pred = shuffle(self.X_word, self.X_year, self.Y_pred)


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
			num_outputs=VOCAB_DIM,
		)

		self.loss = tf.losses.sparse_softmax_cross_entropy(
			labels=self.Y_pred,
			logits=self.Y,
		)

		self.train_step = tf.train.AdamOptimizer(LR).minimize(self.loss)

	def train(self, train_data, dev_data):
		# for i in xrange(N_EPOCHS):
		# 	train_data.shuffle()
		# 	for j in xrange(0, len(train_data) - BATCH_SIZE, BATCH_SIZE):
		# 		batch
		# TODO

if __name__ == "__main__":

	#  TODO #
	# build train, test datasets
	# /TODO #

	model = TemporalLanguageModel()
	model.add_graph()
	# model.train(train_data, dev_data)