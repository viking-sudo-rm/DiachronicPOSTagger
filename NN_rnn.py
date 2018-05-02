from __future__ import division

import tensorflow as tf
import numpy as np
import os, sys
from tensorflow.contrib import rnn
from NN_data_processing_updated import read_in

# To stack more LSTM layers, just add more sizes to this list
LAYERS = [512]

# Maximum length of sentences
MAX_LEN = 32 # changed from 50

# Size of input embedding vectors
EMBED_DIM = 300

# Number of parts of speech
N_POS = 423

EMBED_PATH = "FILL IN" # Set None to disable 
TRAIN_PATH = "/home/accts/gfs22/LING_380/Data/10000/Train"
DEV_PATH = "/home/accts/gfs22/LING_380/Data/10000/Dev"
TEST_PATH = "/home/accts/gfs22/LING_380/Data/10000/Test"

# Year embedding params
START_YEAR = 1800
END_YEAR = 2020

# Hyperparameters
LR = 0.001
N_EPOCHS = 30
BATCH_SIZE = 100

def log(content, input_data):
    print input_data 
    content.write(input_data)


class Dataset:

    """ Wrapper class for storing data. """

    def __init__(self, X_word, X_year, Y_label):
        self.X_word = X_word
        self.X_year = X_year
        self.Y_label = Y_label

    @staticmethod
    def load_old(path):

        X_word_path = os.path.join(path, "X_word_10000")
        X_year_path = os.path.join(path, "X_year_10000")
        Y_path = os.path.join(path, "Y_10000")

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

    @staticmethod
    def load(path):

        X_word_path = os.path.join(path, "X_word_arrays.npz")
        X_year_path = os.path.join(path, "X_year_arrays.npz")
        Y_path = os.path.join(path, "Y_arrays.npz")

        X_word_array = np.load(X_word_path)
        X_year_array = np.load(X_year_path)
        Y_array = np.load(Y_path)

        return Dataset(
            X_word_array,
            X_year_array,
            Y_array 
        )

    def shuffle(self):
        permutation = np.random.permutation(self.X_word.shape[0])
        self.X_word = self.X_word[permutation, :, :]
        self.X_year = self.X_year[permutation, ]
        self.Y_label = self.Y_label[permutation, :]

    def iter_batches(self):
        for i in xrange(0, len(self.X_word) - BATCH_SIZE, BATCH_SIZE):
            yield (
                i, 
                self.X_word[i:i+BATCH_SIZE, :, :],
                self.X_year[i:i+BATCH_SIZE, ],
                self.Y_label[i:i+BATCH_SIZE, :],
            )

    def get_n_batches(self):
        return len(self.X_word) // BATCH_SIZE

    def save(self, savepath):

        X_word_name = "X_word_arrays.npz"
        X_year_name = "X_year_arrays.npz"
        Y_name = "Y_arrays.npz"

        with open(os.path.join(savepath, X_word_name), "wb") as fh:
            np.save(fh, self.X_word)

        with open(os.path.join(savepath, X_year_name), "wb") as fh:
            np.save(fh, self.X_year)

        with open(os.path.join(savepath, Y_name), "wb") as fh:
            np.save(fh, self.Y_label)



class TemporalLanguageModel:

    def add_graph(self, embed_data):

        self.X_word = tf.placeholder(tf.int32, [None, MAX_LEN, EMBED_DIM])
        self.X_year = tf.placeholder(tf.int32, [None])
        self.Y_label = tf.placeholder(tf.int32, [None, MAX_LEN])

        # Embed the inputs
        W_embed = tf.Variable(embed_data)
        X_word = tf.nn.embedding_lookup(W_embed, self.X_word)

        # Can do the same thing for years

        X_year = tf.expand_dims(tf.tile(tf.expand_dims(self.X_year, axis=1), [1, MAX_LEN]), axis=2)
        X = tf.concat([X_word, X_year], axis=2)

        # if embed_data is None:
        #     E_year = X_year
        # else:
        #     E_year_W = tf.Variable(embed_data)
        #     tf.nn.embedding_lookup

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

        #itf.argmax(self.Y, axis=2)
        #2 vectors pointwise equal !
        #argmax by position is right thing if each of argmaxes are equal to thing
       # equal = [tf.argmax(self.Y[i], axis=2) == self.Y_label[i] for i in range(0, self.Y.shape[0])]
        equal = tf.equal(tf.cast(tf.argmax(self.Y, axis=2), tf.int32), tf.cast(self.Y_label, tf.int32))
        self.acc = tf.reduce_mean(tf.cast(equal, tf.float32))

        log_p = tf.gather(tf.log(self.Y), self.Y_label)
        self.log_perp = -tf.reduce_mean(log_p)

        self.loss = tf.losses.sparse_softmax_cross_entropy(
            labels=self.Y_label,
            logits=self.Y,
        )

        self.train_step = tf.train.AdamOptimizer(LR).minimize(self.loss)

    def add_summaries(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.acc)

    def train(self, session, train_data, dev_data, test_data):
        saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())
        n_batches = train_data.get_n_batches()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter( "/home/accts/gfs22/LING_380/Data/Output/Train_Summary", session.graph)
        dev_loss = float("inf")
        #no_improve = 0

        train_data = dev_data
        dev_data = test_data

        data_file = "/home/accts/gfs22/LING_380/Data/Output/may_1_run.txt"

        content=open(data_file, "a")

        #batch_list = list(train_data.iter_batches())
        #log(content, str(len(batch_list)))
        #import pdb; pdb.set_trace()

        for i in xrange(N_EPOCHS):
            #with content:
            log(content, "PRE-SHUFFLED\n")
            log(content, str(i))
            train_data.shuffle()
            log(content, "SHUFFLED!!\n")
            loss = 0.
            for j, batch_X_word, batch_X_year, batch_Y_label in train_data.iter_batches():
                d_loss, _, d_merged = session.run([self.loss, self.train_step, merged], feed_dict={
                    self.X_word: batch_X_word,
                    self.X_year: batch_X_year,
                    self.Y_label: batch_Y_label,
                })
                loss += d_loss
                train_writer.add_summary(d_merged, i*n_batches+j)
                log(content, "#{} BATCH : loss={:.3f}\n".format(i, d_loss)) 
            log(content, "#{} TRAIN : loss={:.3f}\n".format(i, loss / n_batches))
           # prev_dev_loss = dev_loss
            # dev_loss, dev_acc, dev_log_perp = session.run([self.loss, self.acc, self.log_perp], feed_dict={
            #     self.X_word: dev_data.X_word,
            #     self.X_year: dev_data.X_year,
            #     self.Y_label: dev_data.Y_label,
            # })
            #over sequence not part of speech!!

            dev_acc = session.run(self.acc, feed_dict={
                self.X_word: dev_data.X_word,
                self.X_year: dev_data.X_year,
                self.Y_label: dev_data.Y_label,
            })

            #early stopping
          #  if dev_loss<prev_dev_loss:
          #      no_improve=0
          #  else:
          #      no_improve+=1

#            if no_improve>3:
 #               break
            log(content, "#{} DEV : acc={:.3f}\n".format(i, dev_acc)) 

        content.close()
        del train_data
        del dev_data
        saver.save(session, "/home/accts/gfs22/LING_380/Data/Output/model_5_1")

    def test(self, test_data):
        sess = tf.Session()
        saver = tf.train.import_meta_graph('/home/accts/gfs22/LING_380/Data/Output/model.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./'))
        test_acc = session.run(self.acc, feed_dict={
                self.X_word: test_data.X_word,
                self.X_year: test_data.X_year,
                self.Y_label: test_data.Y_label,
        })
        print('Test acc: {}'.format(test_acc))
       # predictions, _ = model.predict(test_data, saver)
        return test_acc

    def rate_of_change(self, test_data, input_year):
        sess = tf.Session()
        saver = tf.train.import_meta_graph('/home/accts/gfs22/LING_380/Data/Output/model.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./'))
        zipped_tuples = zip(test_data.X_word, test_data.X_year, test_data.Y_label)

       # sort(test_data.X_year, by=)
        years = [year for year in test_data.X_year]
        year_dif = [input_year-year for year in years]
        new_years = [input_year-diff_year for diff_year in year_dif if real_year>input_year]
        test_acc = session.run(self.acc, feed_dict={
                self.X_word: test_data.X_word,
                self.X_year: new_years,
                self.Y_label: test_data.Y_label,
        })
        print('Test acc: {}'.format(test_acc))
        return test_acc

    def sample_sentence(self, input_sentence):
        sess = tf.Session()
        saver = tf.train.import_meta_graph('/home/accts/gfs22/LING_380/Data/Output/model.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./'))
        years = range(1810, 2010)
        #use data processing functions
    
def main():

    # TRAIN_PATH = "/home/accts/gfs22/LING_380/Data/Extracted"
    # DEV_PATH = "/home/accts/gfs22/LING_380/Data/Extracted"
    # data = Dataset.load_oldls(TRAIN_PATH)

    # data.shuffle()

    # train_X_word = data.X_word[:140000, :32, :]
    # train_X_year = data.X_year[:140000]
    # train_Y = data.Y_label[:140000, :32]

    # dev_X_word = data.X_word[140000:170000, :32, :]
    # dev_X_year = data.X_year[140000:170000]
    # dev_Y = data.Y_label[140000:170000, :32]

    # test_X_word = data.X_word[170000:, :32, :]
    # test_X_year = data.X_year[170000:]
    # test_Y = data.Y_label[170000:, :32]

    # train_data = Dataset(
    #     train_X_word, 
    #     train_X_year,
    #     train_Y 
    # )

    # dev_data = Dataset(
    #     dev_X_word, 
    #     dev_X_year, 
    #     dev_Y
    # )

    # test_data = Dataset(
    #     test_X_word, 
    #     test_X_year,
    #     test_Y 
    # )

    # TRAIN_SAVE_PATH = "/home/accts/gfs22/LING_380/Data/10000/Train"
    # DEV_SAVE_PATH = "/home/accts/gfs22/LING_380/Data/10000/Dev"
    # TEST_SAVE_PATH = "/home/accts/gfs22/LING_380/Data/10000/Test"

    # train_data.save(TRAIN_SAVE_PATH)

    # dev_data.save(DEV_SAVE_PATH)

    # test_data.save(TEST_SAVE_PATH)
 #   train_data = Dataset(
    #    np.random.uniform(size=[100, MAX_LEN, EMBED_DIM]),
  #      np.random.uniform(size=[100, MAX_LEN]),
     #   np.random.uniform(size=[100, MAX_LEN], high=100),
   # )

    # dev_data = Dataset(
    #     np.random.uniform(size=[100, MAX_LEN, EMBED_DIM]),
    #     np.random.uniform(size=[100, MAX_LEN]),
    #     np.random.uniform(size=[100, MAX_LEN], high=100),
    # )

    train_data = Dataset.load(TRAIN_PATH)
    dev_data = Dataset.load(DEV_PATH)
    test_data = Dataset.load(TEST_PATH)

    if EMBED_PATH is not None:
        with open(EMBED_PATH) as fh:
            embed_data = np.load(fh)
    else:
        embed_data = None

    print("Data Loaded!")

    session = tf.Session()

    print("Declare Model")
    model = TemporalLanguageModel()

    print("Adding Graph")
    model.add_graph(embed_data=embed_data)

    print("Adding summaries")
    model.add_summaries()

    print("Training!")
    model.train(session, train_data, dev_data, test_data)

    #print("Testing")
    #model.test(test_data)


if __name__ == "__main__":
    main()