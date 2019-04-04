from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os, sys, argparse
from data_processing import format_word
from tensorflow.contrib import rnn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import statsmodels.api as sm

# To stack more LSTM layers, just add more sizes to this list
LAYERS = [512]

# Maximum length of sentences
MAX_LEN = 32 

# Size of input embedding vectors
EMBED_DIM = 300

# Number of parts of speech
N_POS = 424

#Declare Paths
DATA_PATH = "/home/accts/gfs22/LING_380/Data/Extracted/"
EMBED_PATH = "/home/accts/gfs22/LING_380/Data/Extracted/EMBED_MAT_FINAL_TRIMMED_R.npz" 
TRAIN_SAVE_PATH = "/home/accts/gfs22/LING_380/Data/Full/Train"
DEV_SAVE_PATH = "/home/accts/gfs22/LING_380/Data/Full/Dev"
TEST_SAVE_PATH = "/home/accts/gfs22/LING_380/Data/Full/Test"
MODEL_PATH = "/home/accts/gfs22/LING_380/Data/Output/FINAL_R"
WRITE_TO_PATH = "/home/accts/gfs22/LING_380/Data/Output/FINAL_R.txt"
X_WORD_FILENAME = "X_word_array_FINAL_TRIMMED_R.npz"
X_YEAR_FILENAME = "X_year_array_FINAL_R.npz"
Y_FILENAME = "Y_array_FINAL_R.npz"
LOSS_POINTS_PATH = "/home/accts/gfs22/LING_380/Data/Output/loss_by_batch.txt"
LEX_PATH = "/home/accts/gfs22/LING_380/Data/Embeddings/lexicon.txt"

DIM_RED = PCA

# Year embedding params
START_YEAR = 1810
END_YEAR = 2010
NUM_YEAR = END_YEAR - START_YEAR

# Hyperparameters
LR = 0.001
N_EPOCHS = 1
BATCH_SIZE = 100 
MAX_THRESHOLD = 600000

#Function that will both print output to the terminal and write it to a file
def log(content, input_data):
    print(input_data)
    content.write(input_data)


class Dataset:

    """ Wrapper class for storing data. """
    def __init__(self, X_word, X_year, Y_label):
        self.X_word = X_word
        self.X_year = X_year
        self.Y_label = Y_label

    @staticmethod
    def load(path, split):

        #Load Data from Data Processing to split in Train/Dev/Test
        if split == 0:
            X_word_path = os.path.join(path, X_WORD_FILENAME)
            X_year_path = os.path.join(path, X_YEAR_FILENAME)
            Y_path = os.path.join(path, Y_FILENAME)

        #Load Train/Dev/Test Data
        else: 
            X_word_path = os.path.join(path, "X_word_array_R.npz")
            X_year_path = os.path.join(path, "X_year_array_R.npz")
            Y_path = os.path.join(path, "Y_array_R.npz")

        X_word_array = np.load(X_word_path)
        X_year_array = np.load(X_year_path)
        Y_array = np.load(Y_path)

        #Combined X_word_array, X_year_array, and Y_array to make a Dataset object
        return Dataset(
            X_word_array,
            X_year_array,
            Y_array 
        )

    #Shuffle Data
    def shuffle(self):
        permutation = np.random.permutation(self.X_word.shape[0])
        self.X_word = self.X_word[permutation, :]
        self.X_year = self.X_year[permutation, ]
        self.Y_label = self.Y_label[permutation, :]

    #Iterate Batches of Data
    def iter_batches(self):
        for i in xrange(0, len(self.X_word) - BATCH_SIZE, BATCH_SIZE):
            yield (
                i, 
                self.X_word[i:i+BATCH_SIZE, :],
                self.X_year[i:i+BATCH_SIZE, ],
                self.Y_label[i:i+BATCH_SIZE, :],
            )

    #Get Number of Batches
    def get_n_batches(self):
        return len(self.X_word) // BATCH_SIZE

    #Save Function 
    #Called Separately on Train, Dev, and Test Data
    def save(self, savepath):

        X_word_name = "X_word_array_R.npz"
        X_year_name = "X_year_array_R.npz"
        Y_name = "Y_array_R.npz"

        with open(os.path.join(savepath, X_word_name), "wb") as fh:
            np.save(fh, self.X_word)

        with open(os.path.join(savepath, X_year_name), "wb") as fh:
            np.save(fh, self.X_year)

        with open(os.path.join(savepath, Y_name), "wb") as fh:
            np.save(fh, self.Y_label)



class TemporalLanguageModel:

    #Create Graph Object
    def add_graph(self):

        #Create placeholders for LSTM
        self.X_word = tf.placeholder(tf.int32, [None, MAX_LEN])
        self.X_year = tf.placeholder(tf.int32, [None])
        self.Y_label = tf.placeholder(tf.int32, [None, MAX_LEN])
        self.embedding_matrix = tf.placeholder(tf.float32, [MAX_THRESHOLD, EMBED_DIM])

        #Look up embeddings for each word
        X_word = tf.nn.embedding_lookup(self.embedding_matrix, self.X_word)

        #Create Year Embedding Layer
        new_years = tf.subtract(self.X_year, START_YEAR)
        unembedded_year = tf.tile(tf.expand_dims(new_years, axis=1), [1, MAX_LEN])
        self.year_embed_mat = tf.get_variable(name="year_embed_mat", shape=(NUM_YEAR, EMBED_DIM), initializer=tf.contrib.layers.xavier_initializer())
        embedded_year = tf.nn.embedding_lookup(self.year_embed_mat, unembedded_year)

        #Concatenate X_word and year embedding layer to get one input
        X = tf.concat([X_word, embedded_year], axis=2)

        #Implement RNN
        rnn_layers = [rnn.LSTMCell(size) for size in LAYERS]
        multi_rnn_cell = rnn.MultiRNNCell(rnn_layers)

        H, _ = tf.nn.dynamic_rnn(
            cell=multi_rnn_cell,
            inputs=X, 
            dtype=tf.float32
        )

        #All POS
        self.Y = tf.contrib.layers.fully_connected(
            inputs=H,
            num_outputs=N_POS,
        )
        
        #Calculate Accuracy
        equal = tf.equal(tf.cast(tf.argmax(self.Y, axis=2), tf.int32), tf.cast(self.Y_label, tf.int32))
        self.acc = tf.reduce_mean(tf.cast(equal, tf.float32))
        self.vec_acc = tf.reduce_mean(tf.cast(equal, tf.float32), axis=1)

        #Calculate Log Perplexity
        mask = tf.cast(tf.one_hot(self.Y_label, N_POS), tf.float32)
        p = tf.reduce_sum(tf.nn.softmax(self.Y) * mask, axis=2)
        self.log_perp = -tf.reduce_sum(tf.log(p), axis=1)/MAX_LEN
        self.perp = tf.exp(self.log_perp)

        #Calculate Loss
        self.loss = tf.losses.sparse_softmax_cross_entropy(
            labels=self.Y_label,
            logits=self.Y,
        )

        #Set train_step that uses AdamOptimizer to minimize loss
        self.train_step = tf.train.AdamOptimizer(LR).minimize(self.loss)

        #Create Graph Object
    def add_graph_FF(self):

        #Create placeholders for LSTM
        self.X_word = tf.placeholder(tf.int32, [None, MAX_LEN])
        self.X_year = tf.placeholder(tf.int32, [None])
        self.Y_label = tf.placeholder(tf.int32, [None, MAX_LEN])
        self.embedding_matrix = tf.placeholder(tf.float32, [MAX_THRESHOLD, EMBED_DIM])

        #Look up embeddings for each word
        X_word = tf.nn.embedding_lookup(self.embedding_matrix, self.X_word)

        #Create Year Embedding Layer
        new_years = tf.subtract(self.X_year, START_YEAR)
        unembedded_year = tf.tile(tf.expand_dims(new_years, axis=1), [1, MAX_LEN])
        self.year_embed_mat = tf.get_variable(name="year_embed_mat", shape=(NUM_YEAR, EMBED_DIM), initializer=tf.contrib.layers.xavier_initializer())
        embedded_year = tf.nn.embedding_lookup(self.year_embed_mat, unembedded_year)

        #Concatenate X_word and year embedding layer to get one input
        X = tf.concat([X_word, embedded_year], axis=2)

        #Implement Feed-Foward
        H  = tf.layers.dense(
            inputs = X,
            units = LAYERS[0],
            activation = tf.nn.sigmoid
        )

        #All POS
        self.Y = tf.contrib.layers.fully_connected(
            inputs=H,
            num_outputs=N_POS,
        )
        
        #Calculate Accuracy
        equal = tf.equal(tf.cast(tf.argmax(self.Y, axis=2), tf.int32), tf.cast(self.Y_label, tf.int32))
        self.acc = tf.reduce_mean(tf.cast(equal, tf.float32))
        self.vec_acc = tf.reduce_mean(tf.cast(equal, tf.float32), axis=1)

        #Calculate Log Perplexity
        mask = tf.cast(tf.one_hot(self.Y_label, N_POS), tf.float32)
        p = tf.reduce_sum(tf.nn.softmax(self.Y) * mask, axis=2)
        self.log_perp = -tf.reduce_sum(tf.log(p), axis=1)/MAX_LEN
        self.perp = tf.exp(self.log_perp)

        #Calculate Loss
        self.loss = tf.losses.sparse_softmax_cross_entropy(
            labels=self.Y_label,
            logits=self.Y,
        )

        #Set train_step that uses AdamOptimizer to minimize loss
        self.train_step = tf.train.AdamOptimizer(LR).minimize(self.loss)


    #Function to train model
    def train(self, session, train_data, dev_data, test_data, embed_data):
        saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())
        n_batches = train_data.get_n_batches()
    
        data_file = WRITE_TO_PATH

        content=open(data_file, "a")

        #Loop through epochs
        for i in xrange(N_EPOCHS):
            log(content, "PRE-SHUFFLED\n")
            log(content, str(i))
            train_data.shuffle()
            log(content, "SHUFFLED!!\n")
            loss = 0.

            #Loop through Batches
            for j, batch_X_word, batch_X_year, batch_Y_label in train_data.iter_batches():
                d_loss, _ = session.run([self.loss, self.train_step], feed_dict={
                    self.X_word: batch_X_word,
                    self.X_year: batch_X_year,
                    self.Y_label: batch_Y_label,
                    self.embedding_matrix: embed_data
                })
                loss += d_loss
                if j%5000==0:
                    log(content, "#{}, {}, BATCH : loss={:.3f}\n".format(i, j, d_loss)) 
            log(content, "#{} TRAIN : loss={:.3f}\n".format(i, loss / n_batches))

            #Calculate Accuracy on Development Set at the end of the epoch
            dev_acc = session.run(self.acc, feed_dict={
                self.X_word: dev_data.X_word,
                self.X_year: dev_data.X_year,
                self.Y_label: dev_data.Y_label,
                self.embedding_matrix : embed_data
            })
            log(content, "#{} DEV : acc={:.3f}\n".format(i, dev_acc)) 

        content.close()
        del train_data
        del dev_data

        #Save model
        saver.save(session, MODEL_PATH)



    #Method to produce clustering plot
    def clustering(self):
        #Restore Model
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)

        #Extract year embedding from restored model
        embedded_var = sess.run(self.year_embed_mat)

        #Fit TSNE to year embedding
        reducer = TSNE(n_components=2, metric="cosine") if DIM_RED == TSNE else PCA(n_components=2)
        image = reducer.fit_transform(embedded_var)
        
        #Plot Clusters
        size = 50
        colors = ["r", "g", "b", "y"]
        for i in xrange(0, len(image), size):
            plt.scatter(*zip(*image[i:i+size,:]), c=colors[int(i / size)])

        plt.legend(["1810-1859", "1860-1909", "1910-1959", "1960-2009"])
        plt.title("Year Embedding Clustering by Half Century ({})".format("TSNE" if DIM_RED == TSNE else "PCA"))

        #Show plot
        plt.show()

    def linear_reduction(self):
        #Restore Model
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)

        #Extract year embedding from restored model
        embedded_var = sess.run(self.year_embed_mat)

        #Fit Dimension Reduction to year embedding
        reducer = TSNE(n_components=1, metric="cosine") if DIM_RED == TSNE else PCA(n_components=1)
        image = reducer.fit_transform(embedded_var)
        lin_vals =  image[:, 0]
        
        plt.scatter(range(1810, 2010), lin_vals)
        plt.title("1D {} vs. Year".format("TSNE" if DIM_RED == TSNE else "PCA"))

        #Calculate R squared
        correlation = np.corrcoef(range(1810, 2010), lin_vals)[0,1]
        print(correlation, correlation**2)

        #Show plot
        plt.show()


    def learning_curve(self, file_name):

        #Open loss values by batch 
        with open(file_name) as fh:
            lines = fh.readlines()
        batch_loss = [map(float,line.split()) for line in lines]
        batch, loss = zip(*batch_loss)

        #plot loss values verses batch number
        plt.title("Training Loss vs. Batch")
        plt.plot(batch, loss)
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.show()

    #Calculate Test Accuracy
    def test(self, test_data, embed_data):
        #Restore Model
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)

        #Calculate Test Accuracy 
        test_acc = sess.run(self.acc, feed_dict={
                self.X_word: test_data.X_word,
                self.X_year: test_data.X_year,
                self.Y_label: test_data.Y_label,
                self.embedding_matrix: embed_data
        })
        print('Test acc: {}'.format(test_acc))
        return test_acc

    def sample_sentence(self, X_word_array, Y_array, X_year, embed_data, words):
        #Restore Model
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)

        #Find sentence corresponding to input X_word_array
        sentence = []
        for i in X_word_array:
            sentence.append(words[i])
            if words[i]==".":
                break

        sentence = " ".join(sentence)

        #Get inputs for sample sentence
        years = np.arange(START_YEAR, END_YEAR)
        X_word_array = np.tile(np.expand_dims(X_word_array, axis=0), [NUM_YEAR, 1])
        Y_array = np.tile(np.expand_dims(Y_array, axis=0), [NUM_YEAR, 1])

        #Calculate Entropy for input sentence over all years
        metric = sess.run(self.perp, feed_dict={
            self.X_word: X_word_array,
            self.X_year: years,
            self.Y_label: Y_array,
            self.embedding_matrix: embed_data
        })

        #Print each sentence and year of sentence
        print(sentence)
        print(X_year)

        #Generate Lowess Curve
        lowess = sm.nonparametric.lowess(metric, years, frac=.3)
        lowess_year = list(zip(*lowess))[0]
        lowess_metric = list(zip(*lowess))[1]

        #Plot Perplexity vs. Years
        plt.scatter(years, metric)
        plt.plot(lowess_year, lowess_metric, c="r")
        plt.xlabel("Years")
        plt.ylabel("Perplexity")
        
        #Show plot
        plt.show()

def read_lex():

    #Open Lexicon
    with open(LEX_PATH) as fh:
        lines = fh.readlines()

    words = {0:"UNK"}

    #Fill in words, a dictionary that maps words to actual embeddings
    for line in lines:
        word_list = line.strip().split("\t")
        if len(word_list) == 5:
            wid, _, word, _, _ = word_list[:5]
            word = format_word(word)
            wid = int(wid)
            if wid not in words:
                words[wid] = word

    return words

#Function to Divide data set into train, dev and test
def cut_dataset(data_path):

    #Load Data
    data = Dataset.load(data_path, 0)

    #Set Indices for Cutting
    NUM_SENT = 750000
    train_end= int(0.7*NUM_SENT)
    dev_end = int(0.85*NUM_SENT)

    #Shuffle data
    data.shuffle()

    #Subset train data
    train_X_word = data.X_word[:train_end, :]
    train_X_year = data.X_year[:train_end]
    train_Y = data.Y_label[:train_end, :]

    #Subset dev data
    dev_X_word = data.X_word[train_end:dev_end, :]
    dev_X_year = data.X_year[train_end:dev_end]
    dev_Y = data.Y_label[train_end:dev_end, :]

    #Subset test data
    test_X_word = data.X_word[dev_end:NUM_SENT, :]
    test_X_year = data.X_year[dev_end:NUM_SENT]
    test_Y = data.Y_label[dev_end:NUM_SENT, :]

    #Return train data object
    train_data = Dataset(
        train_X_word, 
        train_X_year,
        train_Y 
    )

    #Return dev data object
    dev_data = Dataset(
        dev_X_word, 
        dev_X_year, 
        dev_Y
    )

    #Return test data object
    test_data = Dataset(
        test_X_word, 
        test_X_year,
        test_Y 
    )

    #Save train, dev, and test data
    train_data.save(TRAIN_SAVE_PATH)
    dev_data.save(DEV_SAVE_PATH)
    test_data.save(TEST_SAVE_PATH)

def main():

    parser = argparse.ArgumentParser(description="Train LSTM POS model.")

    #Cut the Data set
    parser.add_argument("--cut", action="store_true")

    #Do not train
    parser.add_argument("--notrain", action="store_true")

    # Feed-Forward vs. LSTM
    parser.add_argument("--feedforward", action="store_true")

    args = parser.parse_args()

    #Create Train, Dev, and Test Data
    if args.cut:
        cut_dataset(DATA_PATH)

    #Load Train, Dev, and Test Data
    if not args.notrain:
        train_data = Dataset.load(TRAIN_SAVE_PATH, 1)
        dev_data = Dataset.load(DEV_SAVE_PATH, 1)

    test_data = Dataset.load(TEST_SAVE_PATH, 1)

    #Load Embedding Matrix
    with open(EMBED_PATH) as fh:
        embed_data = np.load(fh)

    #Create a session
    session = tf.Session()

    print("Declare Model")
    model = TemporalLanguageModel()

    print("Adding Graph")
    if(args.feedforward):
        print("Feedforward")
        model.add_graph_FF()
    else:
        print("LSTM")
        model.add_graph()

    if not args.notrain:
        print("Training!")
        model.train(session, train_data, dev_data, test_data, embed_data)

    print("Testing")
    model.test(test_data, embed_data)

    print("Learning Curve")
    model.learning_curve(LOSS_POINTS_PATH)

    print("Linear Reduction")
    model.linear_reduction()

    print("Clustering")
    model.clustering()

    print("Sample Sentence")
    
    #Sample random 15 sentences from test data
    NUM_SENT = 750000
    words = read_lex()
    sample_indices = np.random.uniform(low=0, high=int(0.15*NUM_SENT)-1, size=15).astype(np.int32)
   
    #Run sample_sentence code on data from each of 15 sentences from test data
    for index in sample_indices:
        print(index)
        X_word_arr = test_data.X_word[index, :]
        Y_arr = test_data.Y_label[index, :]
        X_year = test_data.X_year[index]
        model.sample_sentence(X_word_arr, Y_arr, X_year, embed_data, words)


if __name__ == "__main__":
    main()
