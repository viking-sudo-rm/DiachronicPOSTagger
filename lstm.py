# Imports Packages
from __future__ import division
from __future__ import print_function

import argparse
from collections import defaultdict
from data_processing import format_word
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os 
import pickle
import random 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import statsmodels.api as sm
import sys
import tensorflow as tf
from tensorflow.contrib import rnn

# LSTM Layers -- to stack more LSTM layers, just add more sizes to this list
LAYERS = [512]

# Maximum length of sentences
MAX_SENT_LENGTH = 50 

# Size of input embedding vectors
EMBED_DIM = 300

# Number of part of speech tags
N_POS = 424

# Number of sentences in combined train / test data set 
NUM_SENT = 1000000

# Number of sample sentences to examine
NUM_SAMPLE_SENT = 1000

global MODEL_PATH

# Declares paths

# Location where X_word, X_year, Y, and embedding matrix data is stored 
DATA_PATH = "/home/accts/gfs22/DiachronicPOSTagger/Data/Processed_Data/"

# Location where lexicon data is stored
LEX_PATH = "/home/accts/gfs22/DiachronicPOSTagger/Data/Given_Data/Embeddings/lexicon.txt"

# Location where train / test data subsets are stored 
TRAIN_SAVE_PATH = "/home/accts/gfs22/DiachronicPOSTagger/Data/Train"
TEST_SAVE_PATH = "/home/accts/gfs22/DiachronicPOSTagger/Data/Test"

# Locations where the trained model information is stored
MODEL_PATH = "/home/accts/gfs22/DiachronicPOSTagger/All_Models/Model"

# Names of X_word, X_year and Y files 
X_WORD_FILENAME = "X_word_array.npz"
X_YEAR_FILENAME = "X_year_array.npz"
Y_FILENAME = "Y_array.npz"
EMBED_FILENAME = "embed_mat.npz"

# Locations where plots are saved
PLOTS_PATH = "/home/accts/gfs22/DiachronicPOSTagger/Plots/"

# Year embedding parameters
START_YEAR = 1810
END_YEAR = 2010
NUM_YEAR = END_YEAR - START_YEAR

# Hyperparameters
LR = 0.001
N_EPONGS = 1
BATCH_SIZE = 100 
MAX_THRESHOLD = 600000

class Dataset:

    """ Wrapper class for storing data. """
    def __init__(self, X_word, X_year, Y_label):
        self.X_word = X_word
        self.X_year = X_year
        self.Y_label = Y_label

    @staticmethod
    def load(path):
        """
        parameters:
            path: a string, the path where processed X_word, X_year, and Y data is stored
        return: 
            data: a Dataset object, contains loaded X_word, X_year, and Y data
        Loads X_word, X_year, and Y data and returns it as a Dataset object.
        """
        X_word_path = os.path.join(path, X_WORD_FILENAME)
        X_year_path = os.path.join(path, X_YEAR_FILENAME)
        Y_path = os.path.join(path, Y_FILENAME)
        
        X_word_array = np.load(X_word_path)
        X_year_array = np.load(X_year_path)
        Y_array = np.load(Y_path)

        # Combines X_word_array, X_year_array, and Y_array to make a Dataset object
        data =  Dataset(
            X_word_array,
            X_year_array,
            Y_array 
        )

        return data

    def shuffle(self):
        """
        Shuffles X_word, X_year and Y_label components of Dataset object with same ordering. 
        """
        permutation = np.random.permutation(self.X_word.shape[0])
        self.X_word = self.X_word[permutation, :]
        self.X_year = self.X_year[permutation, ]
        self.Y_label = self.Y_label[permutation, :]

    def iter_batches(self):
        """
        Iterates and obtains batches of X_word, X_year, and Y_label data. 
        """
        for i in xrange(0, len(self.X_word) - BATCH_SIZE, BATCH_SIZE):
            yield (
                i, 
                self.X_word[i:i+BATCH_SIZE, :],
                self.X_year[i:i+BATCH_SIZE, ],
                self.Y_label[i:i+BATCH_SIZE, :],
            )

    def get_n_batches(self):
        """
        Returns number of batches.
        """
        return len(self.X_word) // BATCH_SIZE

    def save(self, savepath):
        """
        parameters:
            savepath: a string, location to save data
        Saves X_word, X_year, and Y_label data. This function is called separately on train and test data.
        """
        with open(os.path.join(savepath, X_WORD_FILENAME), "wb") as fh:
            np.save(fh, self.X_word)

        with open(os.path.join(savepath, X_YEAR_FILENAME), "wb") as fh:
            np.save(fh, self.X_year)

        with open(os.path.join(savepath, Y_FILENAME), "wb") as fh:
            np.save(fh, self.Y_label)


class TemporalLanguageModel:

    def add_graph(self, noyear=False, feedforward=False):
        """
        parameters:
            noyear: a boolean, indicates whether year information is included as input to the model
            feedforward: a boolean, indicates whether the model is a feedforward neural network or an LSTM
        Creates a graph for the model. Generates placeholders for X_word, X_year, Y_label, and the embedding matrix. Creates
        year embedding. Details model architecture. Calculates accuracy, log perplexity, and loss. Optimizes network based on loss.
        """
        # Creates placeholders for LSTM
        self.X_word = tf.placeholder(tf.int32, [None, MAX_SENT_LENGTH])
        self.X_year = tf.placeholder(tf.int32, [None])
        self.Y_label = tf.placeholder(tf.int32, [None, MAX_SENT_LENGTH])
        self.embedding_matrix = tf.placeholder(tf.float32, [MAX_THRESHOLD, EMBED_DIM])

        # Looks up embeddings for each word
        X_word = tf.nn.embedding_lookup(self.embedding_matrix, self.X_word)

        # Creates year embedding
        new_years = tf.subtract(self.X_year, START_YEAR)
        unembedded_year = tf.tile(tf.expand_dims(new_years, axis=1), [1, MAX_SENT_LENGTH])
        self.year_embed_mat = tf.get_variable(name="year_embed_mat", shape=(NUM_YEAR, EMBED_DIM), initializer=tf.contrib.layers.xavier_initializer())

        embedded_year = tf.nn.embedding_lookup(self.year_embed_mat, unembedded_year)
        if noyear:
            embedded_year = tf.zeros_like(embedded_year)

        # Concatenates X_word and year embedding to get single combined input
        X = tf.concat([X_word, embedded_year], axis=2)

        if feedforward:
            # Implements Feed-Forward
            H  = tf.layers.dense(
                inputs = X,
                units = LAYERS[0],
                activation = tf.nn.sigmoid
            )

        else:
            # Implements LSTM
            rnn_layers = [rnn.LSTMCell(size) for size in LAYERS]
            multi_rnn_cell = rnn.MultiRNNCell(rnn_layers)

            H, _ = tf.nn.dynamic_rnn(
                cell=multi_rnn_cell,
                inputs=X, 
                dtype=tf.float32
            )

        # POS tags
        self.Y = tf.contrib.layers.fully_connected(
            inputs=H,
            num_outputs=N_POS,
        )
        
        # Calculates accuracy
        equal = tf.equal(tf.cast(tf.argmax(self.Y, axis=2), tf.int32), tf.cast(self.Y_label, tf.int32))
        self.acc = tf.reduce_mean(tf.cast(equal, tf.float32))
        self.vec_acc = tf.reduce_mean(tf.cast(equal, tf.float32), axis=1)

        # Calculates perplexity
        mask = tf.cast(tf.one_hot(self.Y_label, N_POS), tf.float32)
        p = tf.reduce_sum(tf.nn.softmax(self.Y) * mask, axis=2)
        self.log_perp = -tf.reduce_sum(tf.log(p), axis=1)/MAX_SENT_LENGTH
        self.perp = tf.exp(self.log_perp)

        # Calculates loss
        self.loss = tf.losses.sparse_softmax_cross_entropy(
            labels=self.Y_label,
            logits=self.Y,
        )

        # Sets train_step that uses AdamOptimizer to minimize loss
        self.train_step = tf.train.AdamOptimizer(LR).minimize(self.loss)

    def train(self, session, train_data, embed_data):
        """
        parameters:
            session: a TensorFlow session 
            train_data: a Dataset object, includes train X_word, X_year, and Y_label data
            embed_data: a matrix of integers, a matrix of the word embeddings where each row corresponds to a unique word
        Trains model. Reports training loss. Saves model.
        """
        # Sets a seed
        random.seed(2)

        # Generates saver object to save model
        saver = tf.train.Saver()

        # Runs session
        session.run(tf.global_variables_initializer())

        # Calculates number of batches 
        n_batches = train_data.get_n_batches()
    
        # Generates txt file for model output
        model_output = os.path.join(MODEL_PATH, "model_output.txt")

        content=open(model_output, "a")

        # Loops through epochs
        for i in xrange(N_EPOCHS):
            print("PRE-SHUFFLED Train Data")
            print(str(i))
            train_data.shuffle()
            print("SHUFFLED Train Data")
            sys.stdout.flush()
            loss = 0.

            # Loops through Batches
            for j, batch_X_word, batch_X_year, batch_Y_label in train_data.iter_batches():

                # Completes backpropagation 
                d_loss, _ = session.run([self.loss, self.train_step], feed_dict={
                    self.X_word: batch_X_word,
                    self.X_year: batch_X_year,
                    self.Y_label: batch_Y_label,
                    self.embedding_matrix: embed_data
                })
                loss += d_loss
                if j%5000==0:
                    print("#{}, {}, BATCH : loss={:.3f}".format(i, j, d_loss))
                    sys.stdout.flush() 
            print("#{} TRAIN : loss={:.3f}".format(i, loss / n_batches))

        content.close()
        del train_data

        # Saves model
        saver.save(session, MODEL_PATH)

    def calculate_acc(self, corpus_data, embed_data, train):
        """
        parameters:
            corpus_data: a Dataset object, includes X_word, X_year, and Y_label data for the portion of the data set on which 
            accuracy is being evaluated
            embed_data: a matrix of integers, a matrix of the word embeddings where each row corresponds to a unique word
            train: a boolean, indicates whether train or test accuracy is being calculated
        Evaluates the POS tagging accuracy of the model on "corpus_data".
        """
        # Restores model
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)

        # Calculates train accuracy 
        # Due to the large size of the train data set, we divide the data set into 10 parts, calculate the accuracy on each part, 
        # and ultimately average these values.
        if train:
            test_acc_l= []
            bucket = int(corpus_data.X_word.shape[0]/10)
            for i in range(10):
                test_acc = sess.run(self.acc, feed_dict={
                        self.X_word: corpus_data.X_word[bucket*i:bucket*(i+1), :],
                        self.X_year: corpus_data.X_year[bucket*i:bucket*(i+1)],
                        self.Y_label: corpus_data.Y_label[bucket*i:bucket*(i+1), :],
                        self.embedding_matrix: embed_data
                })
                test_acc_l.append(test_acc)
                sys.stdout.flush()
            avg_acc = np.mean(test_acc_l)

            print('Train acc: {}'.format(avg_acc))
            sys.stdout.flush()

        # Calculates test accuracy
        else: 
            test_acc = sess.run(self.acc, feed_dict={
                self.X_word: corpus_data.X_word,
                self.X_year: corpus_data.X_year,
                self.Y_label: corpus_data.Y_label,
                self.embedding_matrix: embed_data})

            print('Test acc: {}'.format(test_acc))
            sys.stdout.flush()

    def linear_reduction(self, feedforward):
        """
        parameters:
            feedforward: a boolean, indicates whether the model is a feedforward neural network or an LSTM 
        Performs principle component analysis on the year embeddings. Calculates the correlation between the first principle 
        component of the year embedding and the sequence of data set years (1810 to 2009). Plots the first principle component
        of the year embeddings against the sequence of data set years. 
        """
        model_type = "FF" if feedforward else "LSTM"

        # Restores Model
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)

        # Extracts year embedding from restored model
        embedded_var = sess.run(self.year_embed_mat)

        # Performs PCA on year embedding
        reducer = PCA(n_components=1)
        image = reducer.fit_transform(embedded_var)

        # Extracts 1st principle component
        lin_vals =  image[:, 0]
        
        # Plots 1st principle component vs. years
        plt.scatter(range(1810, 2010), lin_vals)
        plt.title("1D {} vs. Year".format("PCA") + " " + model_type)

        # Calculates R^squared
        correlation = np.corrcoef(range(1810, 2010), lin_vals)[0,1]
        print(correlation, correlation**2)

        # Saves figure
        plt.savefig(PLOTS_PATH + "Year" + model_type + "/PCA.png")

    def average_perplexity(self, test_data, embed_data, feedforward, loadperplex, yearbucket):
        """
        parameters:
            test_data: a Dataset object, includes X_word, X_year, and Y_label data for the test data
            embed_data: a matrix of integers, a matrix of the word embeddings where each row corresponds to a unique word
            feedforward: a boolean, indicates whether the model is a feedforward neural network or an LSTM
            loadperplex: a boolean, indicates whether to load previously calculated perplexity values of
            each sentence at each year or to calculate these values
            yearbucket: a boolean, indicates whether to load use decades or individual years to aggregate
        For each test data set sentence, calculates the perplexity of the sentence at all years in the data set. Divides the data set 
        into buckets by either year or decade. For each bucket, fits a LOWESS curve where perplexity is a function of year. For all 
        sentences in a bucket, the predicted year of composition is the year corresponding to the minimum of the corresponding 
        perplexity curve. The evaluation metric for each type of model is the average distance across buckets between this 
        predicted year and the bucket's actual middle year. 
        """
        # Verifies model type
        model_type = "FF" if feedforward else "LSTM"

        # Location to save perplexity data 
        pickle_path = PLOTS_PATH + "PickleData" + model_type + "/"

        X_word = test_data.X_word
        X_year = test_data.X_year
        Y_label = test_data.Y_label

        # Verifies bucket type
        bucket_type = ""
        if yearbucket:
            bucket_type = "By_Year"
        else:
            bucket_type = "By_Decade"

        # If loadperplex boolean is false then calculates the perplexity of each sentence at each year
        if not loadperplex:

            # Restores model
            sess = tf.Session()
            saver = tf.train.Saver()
            saver.restore(sess, MODEL_PATH)

            # Initializes dictionaries to store perplexity and years corresponding to each perplexity value
            year_dict = defaultdict(list)
            metric_dict = defaultdict(list)

            # Loops through all test data sentences 
            for idx in range(len(X_word)):
                if idx % 100 == 0:
                    print(idx)
                    sys.stdout.flush()
                sentence = X_word[idx]
                tags = Y_label[idx]

                # Extracts bucket index
                if yearbucket:
                    bucket = X_year[idx]
                else:
                    bucket = (X_year[idx] // 10) * 10

                # Calculates perplexity of sentence at every year 1810 through 2009
                years = np.arange(START_YEAR, END_YEAR)
                X_word_array = np.tile(np.expand_dims(sentence, axis=0), [NUM_YEAR, 1])
                Y_array = np.tile(np.expand_dims(tags, axis=0), [NUM_YEAR, 1])

                metric = sess.run(self.perp, feed_dict={
                    self.X_word: X_word_array,
                    self.X_year: years,
                    self.Y_label: Y_array,
                    self.embedding_matrix: embed_data
                })

                # Stores years 1810 through 2009 in "year_dict" at appropriate bucket key  
                year_dict[bucket].extend(list(years))

                # Stores perplexity of sentence at every year 1810 through 2009 in "metric_dict" at appropriate bucket key
                metric_dict[bucket].extend(list(metric))

            try:
                with open(pickle_path + "year_dict" + bucket_type + ".pkl", "wb") as year_file:
                    pickle.dump(year_dict, year_file)
                with open(pickle_path + "metric_dict" + bucket_type + ".pkl", "wb") as metric_file:
                    pickle.dump(metric_dict, metric_file)
            except Exception as e:
                print(e)

        # If loadperplex boolean is true then loads already calculated perplexity values and corresponding years
        else:
            with open(pickle_path + "year_dict" + bucket_type + ".pkl", "rb") as year_file:
                year_dict = pickle.load(year_file)
            with open(pickle_path + "metric_dict" + bucket_type + ".pkl", "rb") as metric_file:
                metric_dict = pickle.load(metric_file)

        # Evaluates model performance on temporal prediction
        dist_from_actual_year = []

        # Loops through each bucket
        # For each bucket, calculates the absolute distance between the middle year of the bucket and the minimum of the 
        # corresponding LOWESS curve 
        for bucket in year_dict.keys():
            
            # Fits LOWESS curve to all data from bucket
            metric_list = metric_dict[bucket]
            year_list = year_dict[bucket]
            lowess = sm.nonparametric.lowess(metric_list, year_list, frac=.3)
            lowess_year = list(zip(*lowess))[0]
            lowess_metric = list(zip(*lowess))[1]

            # Calculates absolute distance between bucket predicted year and actual middle year
            min_idx, min_met = min((tup for tup in enumerate(lowess_metric)), key=lambda tup: tup[1])
            min_year = lowess_year[min_idx]
            if yearbucket:
                actual_year = bucket
            else:
                actual_year = bucket + 5 
            dist = np.abs(min_year - actual_year)
            dist_from_actual_year.append(dist)

            # Plots bucket perplexity vs. years
            plt.figure()
            plt.plot(lowess_year, lowess_metric, c="r")
            plt.scatter(min_year, min_met)
            plt.annotate(str(min_year), (min_year, min_met))
            plt.xlabel("Years")
            plt.ylabel("Perplexity")
            plt.title(str(bucket) + " "+ model_type)
        
            # Saves plot
            plt.savefig(PLOTS_PATH + "Year" + model_type + "/" + str(bucket_type) + "/" + str(bucket) + ".png")
            print("SAVED")
            print(bucket)
            sys.stdout.flush()

        # Calculates average distance across buckets between bucket predicted year and actual middle year
        print("Mean Dist from Actual Year")
        print(np.mean(dist_from_actual_year))
        sys.stdout.flush()

    def perplexity_sample_sentence(self, X_word_array, actual_year, Y_array, embed_data, word_dict):
        """
        parameters:
            X_word_array: a matrix of integers, each row is identical and corresponds to the indices of the embeddings of
            each word in the given sentence. There are 200 identical rows as having a row for each data set year aids perplexity 
            calculations.
            actual_year: an integer, the year of composition of the given sentence
            Y_array: a matrix of integers, each row is identical and corresponds to the label encoded POS tags of each word in the 
            given sentence. There are 200 identical rows as having a row for each data set year aids perplexity calculations.
            embed_data: a matrix of integers, a matrix of the word embeddings where each row corresponds to a unique word
            word_dict: a dictionary with string keys and integer values, maps word strings (keys) to actual embeddings through 
            embedding IDs (values)
        return:
            dist_year: an integer, the distance between the given sentence's predicted and actual years of composition
            predicted_year: an integer, the predicted year of composition of the given sentence
            actual_year: an integer, the actual year of composition of the given sentence
            sentence: a string, the sequence of words in the given sentence
        For a single sentence, calculates the perplexity of the sentence at all years in the data set (1810 to 2009). Takes the
        predicted year of composition for the sentence to be the year with the minimum perplexity. Returns the sentence's words,
        actual year of composition, predicted year of composition, and the difference between its actual and predicted years
        of composition. 
        """
        # Restores model
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)

        # Finds sequence of word strings corresponding to input "X_word_array"
        sentence = []
        for i in X_word_array:
            sentence.append(word_dict[i])
            if word_dict[i]==".":
                break

        sentence = " ".join(sentence)

        # Calculates perplexity for input sentence across all years
        years = np.arange(START_YEAR, END_YEAR)
        X_word_array = np.tile(np.expand_dims(X_word_array, axis=0), [NUM_YEAR, 1])
        Y_array = np.tile(np.expand_dims(Y_array, axis=0), [NUM_YEAR, 1])

        metric = sess.run(self.perp, feed_dict={
            self.X_word: X_word_array,
            self.X_year: years,
            self.Y_label: Y_array,
            self.embedding_matrix: embed_data
        })

        # Calculates predicted_year of composition and error (distance between predicted and actual year of composition)
        predicted_year = years[np.argmin(metric)]
        dist_year = np.abs(predicted_year - actual_year)
        return dist_year, predicted_year, actual_year, sentence

    def find_sample_sentences(self, test_data, embed_data):
        """
        parameters:
            test_data: a Dataset object, includes X_word, X_year, and Y_label data for the test data
            embed_data: a matrix of integers, a matrix of the word embeddings where each row corresponds to a unique word
        return: 
            dist_year_sorted: a list of integers, the errors for each sentence in increasing order
            predicted_year_sorted: a list of integers, the predicted years of composition of each sentence (ordered by increasing error)
            actual_year_sorted: a list of integers, the actual years of composition of each sentence (ordered by increasing error)
            sentence_sorted: a list of strings, the sentences (ordered by increasing error)
            indices_sorted: a list of integers, the test data set indices of the sentences (ordered by increasing error)
        Samples NUM_SAMPLE_SENT sentences. Generates lists of the predicted years, actual years, sentences, test data set indices,
        and LSTM errors for these sentences. These lists are sorted by error (in increasing order).
        """
        # Generates a dictionary that maps word strings to embedding IDs
        word_dict = generate_word_dict()
        
        X_word_array = test_data.X_word
        actual_years = test_data.X_year
        Y_array = test_data.Y_label
        
        # Initializes predicted year, actual year, sentence, and error (distance between predicted and actual year) lists
        dist_year_l = []
        predicted_year_l = []
        actual_year_l = []
        sent_l = []
        indices_l = []

        # Loops through NUM_SAMPLE_SENT sample sentences
        for idx in range(NUM_SAMPLE_SENT):

            if idx%25 == 0:
                print(idx)
                sys.stdout.flush()

            # Calls "perplexity_sample_sentence" function on each sentence
            X_year = actual_years[idx]
            X_word = X_word_array[idx] 
            Y = Y_array[idx]
            dist_year, predicted_year, actual_year, sentence = self.perplexity_sample_sentence(X_word, X_year, Y, embed_data, word_dict)
        
            # For sentences longer than 5 words keeps track of the predicted year, actual year, sentence, index, and error 
            sent_words = sentence.strip().split(" ")
            if len(sent_words) > 5:
                dist_year_l.append(dist_year)
                predicted_year_l.append(predicted_year)
                actual_year_l.append(actual_year)
                sent_l.append(sentence)
                indices_l.append(idx)

        tuples = zip(dist_year_l, predicted_year_l, actual_year_l, sent_l, indices_l)
        tuples.sort(key = lambda tuple: tuple[0])
        dist_year_sorted, predicted_year_sorted, actual_year_sorted, sentence_sorted, indices_sorted = zip(*tuples)
        return dist_year_sorted, predicted_year_sorted, actual_year_sorted, sentence_sorted, indices_sorted

    def find_minimum_sample_sentences(self, test_data, embed_data, dist_year_sorted, predicted_year_sorted, actual_year_sorted, sentence_sorted, indices_sorted):
        """
        parameters:
            test_data: a Dataset object, includes X_word, X_year, and Y_label data for the test data
            embed_data: a matrix of integers, a matrix of the word embeddings where each row corresponds to a unique word
            dist_year_sorted: a list of integers, the errors for each sentence in increasing order
            predicted_year_sorted: a list of integers, the predicted years of composition of each sentence (ordered by increasing error)
            actual_year_sorted: a list of integers, the actual years of composition of each sentence (ordered by increasing error)
            sentence_sorted: a list of strings, the sentences (ordered by increasing error)
            indices_sorted: a list of integers, the test data set indices of the sentences (ordered by increasing error)
        Prints the sentence string, LSTM error (distance between predicted and actual year of composition), predicted year of 
        composition, actual year of composition, and feedforward error for the 10 sentences of those sampled that have the 
        smallest LSTM errors.
        """
        # Generates a dictionary that maps word strings to embedding IDs
        word_dict = generate_word_dict()

        # Restores model
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)

        # Loops through 10 sentences with smallest LSTM errors
        for i in range(10):
            print("Sentence # " + str(i+1))
            sys.stdout.flush()
            print("LSTM Error")
            print(dist_year_sorted[i])
            print("Predicted Year")
            print(predicted_year_sorted[i])
            print("Actual Year")
            print(actual_year_sorted[i])
            print("Sentence")
            print(sentence_sorted[i])

            # Calculates feedforward error
            idx = indices_sorted[i]
            X_year = test_data.X_year[idx]
            X_word_array = test_data.X_word[idx] 
            Y_array = test_data.Y_label[idx]
            sys.stdout.flush()
            dist_year, predicted_year, actual_year, sentence = self.perplexity_sample_sentence(X_word_array, X_year, Y_array, embed_data, word_dict)
            print("Feedforward Error")
            print(dist_year)
            sys.stdout.flush()
            print(" ")

def generate_word_dict():
    """
    return:
        word_dict: a dictionary with string keys and integer values, maps word strings (keys) to actual embeddings through 
        embedding IDs (values)
    Uses lexicon to create "word_dict", a dictionary that maps string words to embedding IDs.
    """
    # Opens Lexicon
    with open(LEX_PATH) as fh:
        lines = fh.readlines()

    # Initializes 0 index to "UNK" symbol for unknown words
    words = {0:"UNK"}

    # Initializes dictionary with integer word IDs as keys and string words as values
    for line in lines:
        word_list = line.strip().split("\t")
        if len(word_list) == 5:
            wid, _, word, _, _ = word_list[:5]
            word = format_word(word)
            wid = int(wid)
            if wid not in words:
                words[wid] = word

    return words

def cut_dataset():
    """
    Divides data into train and test portions and saves these parts of the data set separately.
    """
    # Loads Data
    data = Dataset.load(DATA_PATH)

    # Sets indices for cutting
    train_end= int(0.9*NUM_SENT)

    # Subsets train data
    train_X_word = data.X_word[:train_end, :]
    train_X_year = data.X_year[:train_end]
    train_Y = data.Y_label[:train_end, :]

    # Subsets test data
    test_X_word = data.X_word[train_end:NUM_SENT, :]
    test_X_year = data.X_year[train_end:NUM_SENT]
    test_Y = data.Y_label[train_end:NUM_SENT, :]

    # Returns train Dataset object
    train_data = Dataset(
        train_X_word, 
        train_X_year,
        train_Y 
    )

    # Returns test Dataset object
    test_data = Dataset(
        test_X_word, 
        test_X_year,
        test_Y 
    )

    # Saves train and test data
    train_data.save(TRAIN_SAVE_PATH)
    test_data.save(TEST_SAVE_PATH)

def main():
    global MODEL_PATH, N_EPOCHS

    parser = argparse.ArgumentParser(description="Train LSTM POS model.")

    # Determines whether to cut the data set into train and test parts 
    parser.add_argument("--cut", action="store_true")

    # Determines whether to train a new model or not
    parser.add_argument("--notrain", action="store_true")

    # Determines whether to implement feedforward vs. LSTM model
    parser.add_argument("--feedforward", action="store_true")

    # Determines whether to include year information as input or not
    parser.add_argument("--noyear", action="store_true")

    # Determines whether to load previously calculated perplexity data or to calculate these numbers
    parser.add_argument("--loadperplex", action="store_true")

    args = parser.parse_args()

    # Splits train and test data 
    if args.cut:
        cut_dataset()
        print("Data Cut")
        sys.stdout.flush()

    # Loads train and test data
    train_data = Dataset.load(TRAIN_SAVE_PATH)
    print("Load Train Data")
    sys.stdout.flush()

    test_data = Dataset.load(TEST_SAVE_PATH)
    print("Load Test Data")
    sys.stdout.flush()

    # Loads embedding matrix
    EMBED_PATH = os.path.join(DATA_PATH, EMBED_FILENAME)
    with open(EMBED_PATH) as fh:
        embed_data = np.load(fh)

    print("Load Embedding Matrix")
    sys.stdout.flush()

    # Creates a session
    session = tf.Session()

    # Declares a model (either feedforward or LSTM) and modifies MODEL_PATH
    model = TemporalLanguageModel()

    noyear = args.noyear
    feedforward = args.feedforward

    if noyear:
        MODEL_PATH += "_NY"
    if(args.feedforward):
        print("Feedforward")
        MODEL_PATH += "_FF"
    else:
        print("LSTM")

    print("Adding Graph")
    sys.stdout.flush()
    model.add_graph(noyear, feedforward)

    # Trains model
    if not args.notrain:
        print("Training model")
        sys.stdout.flush()
        model.train(session, train_data, embed_data)

    # Evaluates train and test accuracy 
    print("Calculating Accuracy")
    sys.stdout.flush()
    print("Training Accuracy")
    model.calculate_acc(train_data, embed_data, 1)
    sys.stdout.flush()
    print("Testing Accuracy")
    model.calculate_acc(test_data, embed_data, 0)
    sys.stdout.flush()

    if not noyear:

        # Performs PCA analysis
        print("Linear Reduction")
        sys.stdout.flush()
        model.linear_reduction(feedforward)
        sys.stdout.flush()

        # Performs temporal prediction
        print("Average Perplexity")
        sys.stdout.flush()

        # Uses year buckets
        print("By Year")
        model.average_perplexity(test_data, embed_data, feedforward, args.loadperplex, 1)

        # Uses decade buckets 
        print("By Decade")
        model.average_perplexity(test_data, embed_data, feedforward, args.loadperplex, 0)
        sys.stdout.flush()

        # Samples sentences and examines ten best predicted cases
        print("Sample Sentences")
        sys.stdout.flush()

        # Samples 1000 sentences and obtains their predicted year, actual year, sentence, and error (distance between predicted 
        # and actual year) lists
        print("Find Sample Sentences")
        dist_year_sorted, predicted_year_sorted, actual_year_sorted, sentence_sorted, indices_sorted = model.find_sample_sentences(test_data, embed_data)
        
        # Finds the predicted year, actual year, sentence, LSTM error and feedforward errors of the ten best predicted sentences
        print("Select Ten Best Sample Sentences")
        MODEL_PATH += "_FF"
        model_FF = TemporalLanguageModel()
        tf.reset_default_graph()
        model_FF.add_graph(noyear, 1)
        model_FF.find_minimum_sample_sentences(test_data, embed_data, dist_year_sorted, predicted_year_sorted, actual_year_sorted, sentence_sorted, indices_sorted)

if __name__ == "__main__":
    main()
