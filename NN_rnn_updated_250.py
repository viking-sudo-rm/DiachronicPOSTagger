from __future__ import division

import tensorflow as tf
import numpy as np
import os, sys
from tensorflow.contrib import rnn
from NN_data_processing_updated import read_in

# To stack more LSTM layers, just add more sizes to this list
LAYERS = [512]

NUM_YEAR = 200

# Maximum length of sentences
MAX_LEN = 32 # changed from 50

# Size of input embedding vectors
EMBED_DIM = 300

# Number of parts of speech
N_POS = 423

EMBED_PATH = "/home/accts/gfs22/LING_380/Data/Extracted/EMBED_MAT_MOD.npz" 
TRAIN_SAVE_PATH = "/home/accts/gfs22/LING_380/Data/Full/Train"
DEV_SAVE_PATH = "/home/accts/gfs22/LING_380/Data/Full/Dev"
TEST_SAVE_PATH = "/home/accts/gfs22/LING_380/Data/Full/Test"

# Year embedding params
START_YEAR = 1800
END_YEAR = 2020

# Hyperparameters
LR = 0.001
N_EPOCHS = 30
BATCH_SIZE = 150 #100
MAX_THRESHOLD = 200000


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
    def load(path, split):

        if split == 0:
            X_word_path = os.path.join(path, "X_word_array_200000.npz")
            X_year_path = os.path.join(path, "X_year_array_5_1.npz")
            Y_path = os.path.join(path, "Y_array_5_1.npz")

        else: 
            X_word_path = os.path.join(path, "X_word_array_250.npz")
            X_year_path = os.path.join(path, "X_year_array_250.npz")
            Y_path = os.path.join(path, "Y_array_250.npz")

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
        self.X_word = self.X_word[permutation, :]
        self.X_year = self.X_year[permutation, ]
        self.Y_label = self.Y_label[permutation, :]

    def iter_batches(self):
        for i in xrange(0, len(self.X_word) - BATCH_SIZE, BATCH_SIZE):
            yield (
                i, 
                self.X_word[i:i+BATCH_SIZE, :],
                self.X_year[i:i+BATCH_SIZE, ],
                self.Y_label[i:i+BATCH_SIZE, :],
            )

    def get_n_batches(self):
        return len(self.X_word) // BATCH_SIZE

    def save(self, savepath):

        X_word_name = "X_word_array_250.npz"
        X_year_name = "X_year_array_250.npz"
        Y_name = "Y_array_250.npz"

        with open(os.path.join(savepath, X_word_name), "wb") as fh:
            np.save(fh, self.X_word)

        with open(os.path.join(savepath, X_year_name), "wb") as fh:
            np.save(fh, self.X_year)

        with open(os.path.join(savepath, Y_name), "wb") as fh:
            np.save(fh, self.Y_label)



class TemporalLanguageModel:

    def add_graph(self):

        self.X_word = tf.placeholder(tf.int32, [None, MAX_LEN])
        self.X_year = tf.placeholder(tf.int32, [None])
        self.Y_label = tf.placeholder(tf.int32, [None, MAX_LEN])
        self.embedding_matrix = tf.placeholder(tf.float32, [200000, 300])


        #print "EMBED DIM"
        #print embed_data.shape
        # Embed the inputs
        #W_embed = tf.constant(embed_data)
       # place = tf.placeholder(tf.float32, shape=(1728400, 300))
        #X_word = tf.where(self.X_word>MAX_THRESHOLD, tf.zeros(self.X_word.shape), self.X_word)
        X_word = tf.nn.embedding_lookup(self.embedding_matrix, self.X_word)
        #print X_word.shape

        #print tf.shape(self.X_year)
        new_years = tf.subtract(self.X_year, 1810)
        #print tf.shape(new_years)
        unembedded_year = tf.tile(tf.expand_dims(new_years, axis=1), [1, MAX_LEN])
        print tf.shape(unembedded_year)
        #unembedded_year = tf.expand_dims(new_years, axis=1)
        #new matrix
        year_embed_mat = tf.get_variable(name="year_embed_mat", shape=(NUM_YEAR, EMBED_DIM), initializer=tf.contrib.layers.xavier_initializer())
        embedded_year = tf.nn.embedding_lookup(year_embed_mat, unembedded_year)

        # Can do the same thing for years

       # X_year = tf.expand_dims(tf.tile(embedded_year, [1, MAX_LEN]), axis=2)

        # Can do the same thing for years
        print embedded_year.shape
        #X_year = tf.expand_dims(tf.tile(embedded_year, [1, MAX_LEN]), axis=2)
        #X_year = tf.expand_dims(tf.tile(tf.expand_dims(self.X_year, axis=1), [1, MAX_LEN]), axis=2)
       # X_year = tf.expand_dims(tf.tile(self.X_year, [1, MAX_LEN]), axis=2)

        #X = tf.concat([X_word, X_year], axis=2)

       # X_year = tf.expand_dims(tf.tile(tf.expand_dims(self.X_year, axis=1), [1, MAX_LEN]), axis=2)
        X = tf.concat([X_word, embedded_year], axis=2)

        # if embed_data is None:
        #     E_year = X_year
        # else:
        #     E_year_W = tf.Variable(embed_data)
        #     tf.nn.embedding_lookup

        #matrix number of years by embed size - xavier initliazation

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

    def train(self, session, train_data, dev_data, test_data, embed_data):
        saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())
        n_batches = train_data.get_n_batches()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter( "/home/accts/gfs22/LING_380/Data/Output/Train_Summary", session.graph)
        dev_loss = float("inf")
        #no_improve = 0

        data_file = "/home/accts/gfs22/LING_380/Data/Output/may_2_run_250.txt"

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
            #sess.run(embedding_init, feed_dict={self.embedding_matrix: embed_data})
            print np.where(train_data.X_word>MAX_THRESHOLD)
            for j, batch_X_word, batch_X_year, batch_Y_label in train_data.iter_batches():
                d_loss, _, d_merged = session.run([self.loss, self.train_step, merged], feed_dict={
                    self.X_word: batch_X_word,
                    self.X_year: batch_X_year,
                    self.Y_label: batch_Y_label,
                    self.embedding_matrix: embed_data
                })
                loss += d_loss
                train_writer.add_summary(d_merged, i*n_batches+j)
                log(content, "#{}, {}, BATCH : loss={:.3f}\n".format(i, j, d_loss)) 
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
                self.embedding_matrix : embed_data
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
        saver.save(session, "/home/accts/gfs22/LING_380/Data/Output/model_5_2_250")

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

    data_path = "/home/accts/gfs22/LING_380/Data/Extracted/"
#    DEV_PATH = "/home/accts/gfs22/LING_380/Data/Extracted"
    data = Dataset.load(data_path, 0)

    num_sent = data.X_word.shape[0]
    print np.where(data.X_word>MAX_THRESHOLD)
    train_end= 175000
    dev_end = 212500
    data.shuffle()

    train_X_word = data.X_word[:train_end, :]
    train_X_year = data.X_year[:train_end]
    train_Y = data.Y_label[:train_end, :]

    dev_X_word = data.X_word[train_end:dev_end, :]
    dev_X_year = data.X_year[train_end:dev_end]
    dev_Y = data.Y_label[train_end:dev_end, :]

    test_X_word = data.X_word[dev_end:250000, :]
    test_X_year = data.X_year[dev_end:250000]
    test_Y = data.Y_label[dev_end:250000, :]
    #print test_Y.shape

    train_data = Dataset(
        train_X_word, 
        train_X_year,
        train_Y 
    )

    dev_data = Dataset(
        dev_X_word, 
        dev_X_year, 
        dev_Y
    )

    test_data = Dataset(
        test_X_word, 
        test_X_year,
        test_Y 
    )


    train_data.save(TRAIN_SAVE_PATH)

    dev_data.save(DEV_SAVE_PATH)

    test_data.save(TEST_SAVE_PATH)

   #  print np.where(train_data.X_word>MAX_THRESHOLD)
   # print("WEIRDER")

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

   # TRAIN_PATH = "/home/accts/gfs22/LING_380/Data/10000/Train"
   # DEV_PATH = "/home/accts/gfs22/LING_380/Data/10000/Dev"
   # TEST_PATH = "/home/accts/gfs22/LING_380/Data/10000/Test"

    train_data = Dataset.load(TRAIN_SAVE_PATH, 1)
    dev_data = Dataset.load(DEV_SAVE_PATH, 1)
    test_data = Dataset.load(TEST_SAVE_PATH, 1)

    print np.where(train_data.X_word>MAX_THRESHOLD)
    print ("WEIRD")

    if EMBED_PATH is not None:
        with open(EMBED_PATH) as fh:
            embed_data = np.load(fh)
            #embed_data = embed_data[:MAX_THRESHOLD,]
            #print "loaded embed"
    else:
        embed_data = None

  #  print("Data Loaded!")

    print embed_data.shape
    session = tf.Session()

    print("Declare Model")
    model = TemporalLanguageModel()

    print("Adding Graph")
    model.add_graph()

    print("Adding summaries")
    model.add_summaries()

    print("Training!")
    model.train(session, train_data, dev_data, test_data, embed_data)

    #print("Testing")
    #model.test(test_data)


if __name__ == "__main__":
    main()