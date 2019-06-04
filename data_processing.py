# Imports packages
from gensim.models import KeyedVectors 
import numpy as np 
import os
import pickle as pk
import re
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
import sys
import tensorflow as tf 

# Declares paths

# Location where all data is stored
DATA_PATH = "/home/accts/gfs22/DiachronicPOSTagger/Data/"

# Location where embedding data is stored
EMBED_PATH = os.path.join(DATA_PATH, "Given_Data/Embeddings/word2vec/data")

# Location where corpus data is stored
CORPUS_PATH = os.path.join(DATA_PATH, "Given_Data/Corpus")

# Location where lexicon data is stored
LEX_PATH = os.path.join(DATA_PATH, "Given_Data/Embeddings/lexicon.txt")

# Location where processed data is saved
SAVE_PATH = os.path.join(DATA_PATH, "Processed_Data")

# Sets constants

# Number of most common words for which embeddings are stored
NUM_COMMON_WORDS = 600000

# Total number of words in the original data set vocabulary 
NUM_VOCAB_WORDS = 3954340

# Maximum length of any sentence
MAX_SENT_LENGTH = 50

# Size of input embedding vectors
EMBED_DIM = 300

# Number of sentences in corpus that exceed 50 words
NUM_SENT_EXCEED_MAX_LEN = 0

# Number of total sentences in corpus
NUM_SENT_TOTAL = 0

def load_embeddings():
	"""
    return:
        model: a matrix of floats, word2vec GoogleNews embeddings
        
    Loads and returns word2vec embeddings
    """
	embeddings = KeyedVectors.load_word2vec_format(EMBED_PATH, binary=True)
	return embeddings

def format_word(word):
	"""
	parameters:
        word: a string, a word in the corpus
    return:
        modified_word: a string, modified version of "word" that is lower case and does not include dashes 
        
    Returns input "word" with dashes removed and in lower case.
    """
	dashes = r"[- ]*"
	modified_word = re.sub(dashes, "", word.lower())
	return modified_word

def read_lex(embeddings):
	"""
    parameters:
        embeddings: a matrix of floats, word2vec GoogleNews embeddings
    return:
    	le: a label encoder, the label encoder for POS tags
    	POS_tags: a list of strings, the list of all POS tags
    	word_dict: a dictionary with string keys and integer values, maps word strings (keys) to actual embeddings through embedding IDs (values)
    	embed_mat: a matrix of integers, a matrix of the word embeddings where each row corresponds to a unique word
    Uses lexicon to initialize a label encoder for POS, a list of POS tags, a dictionary that maps words to their embeddings, and a matrix of all embeddings. 
    """
	# Opens lexicon
	with open(LEX_PATH) as fh:
		lines = fh.readlines()

	# Initializes embedding matrix to random normally distributed numbers
	embed_mat = np.random.normal(size=(NUM_VOCAB_WORDS, EMBED_DIM))

	# Initializes list of all POS tags
	POS_tags = []

	# Initializes dictionary that maps words to embeddings
	word_dict = {}

	# Fills in embed_mat, POS_tags, and word_dict appropriately
	for line in lines:
		word_list = line.strip().split("\t")
		if len(word_list) == 5:
			wid, _, word, _, pos = word_list[:5]
			word = format_word(word)
			wid = int(wid)
			pos = format_POS(pos)
			if word not in word_dict:
				word_dict[word] = wid
			POS_tags.append(pos)
			if word in embeddings:
				embed_mat[wid, :] = embeddings[word]
		else:
			POS_tags.append("N/A")

	# Fits label encoder to POS tags
	le = LabelEncoder()
	le.fit(POS_tags)

	return le, POS_tags, word_dict, embed_mat

def read_single_file(file_name, le, POS_tags, word_dict):
	"""
    parameters:
        file_name: a string, the name of the file under examination 
        le: a label encoder, the label encoder for POS tags
    	POS_tags: a list of strings, the list of all POS tags
    	word_dict: a dictionary with string keys and integer values, maps word strings (keys) to actual embeddings through embedding ids (values)
    return:
		X_word_array: a matrix of integers, each row corresponds to a list of the indices of the embeddings of each word in a sentence in the 
		document
		X_year_array: a list of integers, repeats the year of composition for each word in the sentence for each sentence in the document
		Y_array: a matrix of integers, each row corresponds to a list of the label encoded POS tags of each word in a sentence in the document
        
    For a given document generates matrix corresponding to indices for word embeddings (X_word), list corresponding to year of composition of each
    sentence (X_year_array), and matrix of label encoded POS tags of each word in a sentence in the document (Y_array).
    """
    # Variable corresponding to number of sentences that contain more than 50 words
	global NUM_SENT_EXCEED_MAX_LEN

	global NUM_SENT_TOTAL

	# Words to ignore
	ignore = {"@", "--", "-- ", " --"}

	# Extracts the Year
	split_fn = file_name.split("_")
	X_year = int(split_fn[-2])

	# Opens file
	with open(file_name) as fh:
		lines = fh.readlines()

	# Variables to store embedding indices and label encoded POS tags 
	sent_X_word = []
	sent_Y = []
	X_word = np.zeros(shape=(MAX_SENT_LENGTH,), dtype=np.int32)
	Y = np.zeros(shape=(MAX_SENT_LENGTH, ), dtype=np.int32)

	# Number of words in current sentence
	num_words_cur_sent = 0

 	# Loops through line of document
	for i, line in enumerate(lines):

		word_list = line.strip().split("\t")

		if len(word_list)==3 and word_list[0] not in ignore:

			# Extracts and formats word and POS
			word, lemma, pos = word_list[:3]
			pos = format_POS(pos)
			word = format_word(word)

			# Removes words that begin with @ and words not in vocabulary
			if word.startswith("@") or word not in word_dict:
				continue

			# Fills X_word with indices for word embeddings and Y with POS tags for each word
			if num_words_cur_sent < MAX_SENT_LENGTH:
				X_word[num_words_cur_sent] = word_dict[word]
				Y[num_words_cur_sent] = le.transform([pos])[0]

			# Begins a new sentence
			if lemma == '.':
				sent_X_word.append(X_word)
				X_word = np.zeros(shape=(MAX_SENT_LENGTH,), dtype=np.int32)
				sent_Y.append(Y)
				Y = np.zeros(shape=(MAX_SENT_LENGTH, ), dtype=np.int32)
				num_words_cur_sent = 0

			# Iterates num_words_cur_sent if we are not beginning a new sentence
			else:
				num_words_cur_sent += 1

				# Checks if sentence exceeds maximum sentence length
				if num_words_cur_sent == MAX_SENT_LENGTH:
					NUM_SENT_EXCEED_MAX_LEN = NUM_SENT_EXCEED_MAX_LEN + 1


	# If we have sentences in a document, joins data from all sentences to make combined arrays
	if sent_X_word != []:
		X_word_array = np.stack(sent_X_word, axis=0)
		X_year_array = np.repeat(X_year, len(sent_X_word))
		Y_array = np.stack(sent_Y, axis=0)
		print(X_word_array.shape[0])
		print(NUM_SENT_TOTAL)
		NUM_SENT_TOTAL = NUM_SENT_TOTAL + X_word_array.shape[0]
		print(NUM_SENT_TOTAL)
		return X_word_array, X_year_array, Y_array

	# If there are no sentences in a document returns all None values.
	else:
		return None, None, None

# Formats POS by taking all information before the first underscore -- generates 423 unique POS tags
format_POS = lambda pos: pos.split("_")[0]

def read_all_files(le, POS_tags, word_dict):
	"""
	parameters:
        le: a label encoder, the label encoder for POS tags
    	POS_tags: a list of strings, the list of all POS tags
    	word_dict: a dictionary with string keys and integer values, maps word strings (keys) to actual embeddings through embedding IDs (values)
    return:
		X_word_array: a matrix of integers, each row corresponds to a list of the indices of the embeddings of each word in a sentence; there is a row 
		for each of the 1,000,000 final sentences
		X_year_array: a list of integers, each entry corresponds to the year of composition of a sentence; there is an entry for each of the 1,000,000 
		final sentences
		Y_array: a matrix of integers, each row corresponds to a list of the label encoded POS tags of each word in a sentence; there is a row 
		for each of the 1,000,000 final sentences
    From all documents selects 1,000,000 sentences. Generates matrix corresponding to indices for word embeddings (X_word), list corresponding to year of 
    composition of each sentence (X_year_array), and matrix of label encoded POS tags (Y_array) for each word for each of these sentences.
    """
	# Variable corresponding to number of sentences that contain more than 50 words
	global NUM_SENT_EXCEED_MAX_LEN

	# Variable corresponding to total number of sentences 
	global NUM_SENT_TOTAL

	# Initializes intermediate variables to store data from all files
	X_word_arrays_total, X_year_arrays_total, Y_arrays_total = [], [], []

	# Loops through all files
	for dirpath, dirnames, filenames in os.walk(CORPUS_PATH):

		# Prints directory and number of files in directory
		print(dirpath)
		print(len(filenames))
		sys.stdout.flush()

		if len(filenames)!=0:
			X_word_arrays, X_year_arrays, Y_arrays = [], [], []

			# Calls "read_single_file" function on each file
			# Appends results from each file to X_word_arrays, X_year_arrays, and Y_arrays
			for i, file in enumerate(filenames):
				if i%100 == 0:
					print(i)
					sys.stdout.flush()
				X_word_array, X_year_array, Y_array = read_single_file(os.path.join(dirpath, file), le, POS_tags, word_dict)
				if X_word_array is not None and X_year_array is not None and Y_array is not None:
					X_word_arrays.append(X_word_array)
					X_year_arrays.append(X_year_array)
					Y_arrays.append(Y_array)

			# Appropriately reshapes X_word_arrays_total, X_year_arrays_total, Y_arrays_total into numpy arrays 
			X_word_array_int = np.concatenate(X_word_arrays, axis=0)
			X_year_array_int = np.concatenate(X_year_arrays, axis=0)
			Y_array_int = np.concatenate(Y_arrays, axis=0)

			# Samples 50,000 sentences from each decade
			indices = range(X_word_array_int.shape[0])
			np.random.shuffle(indices)
			indices = indices[:50000]
			X_word_array_int = X_word_array_int[indices, :]
			X_year_array_int = X_year_array_int[indices]
			Y_array_int = Y_array_int[indices, :]
			X_word_arrays_total.append(X_word_array_int)
			X_year_arrays_total.append(X_year_array_int)
			Y_arrays_total.append(Y_array_int)

	# Reshapes X_word_arrays_total, X_year_arrays_total, and Y_arrays_total
	X_word_array = np.concatenate(X_word_arrays_total, axis=0)
	X_year_array = np.concatenate(X_year_arrays_total, axis=0)
	Y_array = np.concatenate(Y_arrays_total, axis=0)
	
	# Shuffles sentence ordering 
	total_indices = range(X_word_array.shape[0])
	np.random.shuffle(total_indices)
	X_word_array = X_word_array[total_indices, :]
	X_year_array = X_year_array[total_indices]
	Y_array = Y_array[total_indices, :]

	# Prints percentage of sentences in corpus that are longer than 50 words
	print("Percentage Exceed")
	print(NUM_SENT_EXCEED_MAX_LEN)
	print(NUM_SENT_TOTAL)
	print(float(NUM_SENT_EXCEED_MAX_LEN)/NUM_SENT_TOTAL)

	return X_word_array, X_year_array, Y_array

def limit_vocabulary(X_word_array, embed_mat):
	"""
	parameters:
        X_word_array: a matrix of integers, each row corresponds to a list of the indices of the embeddings of each word in a sentence; there is a row 
		for each of the 1,000,000 final sentences
		embed_mat: a matrix of integers, a matrix of the word embeddings where each row corresponds to a unique word
	return:
        X_word_array: a matrix of integers, each row corresponds to a list of the indices of the embeddings of each word in a sentence; there is a row 
		for each of the 1,000,000 final sentences; only stores indices for 600,000 most common words.
		embed_mat: a matrix of integers, a matrix of the word embeddings where each row corresponds to a unique word; only stores embeddings for 600,000 
		most common words.
    Ensures the "X_word_array" only stores indices for the 600,000 most common words and "embed_mat" only has embeddings for the 600,000
    most common words.
    """
	# Only stores indices in X_word for embeddings of 600,000 most common words
	indices = np.where(X_word_array>NUM_COMMON_WORDS)
	X_word_array[indices]=0

	# Only stores embeddings of 600,000 most common words
	embed_mat = embed_mat[:NUM_COMMON_WORDS,]

	return X_word_array, embed_mat

if __name__ == '__main__':

	# Loads embeddings
	embeddings = load_embeddings()
	print("Load Embeddings")
	sys.stdout.flush()

	# Uses lexicon to initialize a label encoder for POS, a list of POS tags, a dictionary that maps words to their embeddings, 
	# and a matrix of all embeddings 
	le, POS_tags, word_dict, embed_mat = read_lex(embeddings)
	print("Load Lexicon")
	sys.stdout.flush()

	# Creates a set with all the POS tags
	POS_tags = set(POS_tags)

	# Generates X_word_array, X_year_array, and Y_array
	X_word_array, X_year_array, Y_array = read_all_files(le, POS_tags, word_dict)

	# Ensures that only 600,000 most common words are considered
	X_word_array, embed_mat = limit_vocabulary(X_word_array, embed_mat)

	# Saves X_year_array
	with open(os.path.join(SAVE_PATH, "X_year_array.npz"), "wb") as fh:
		np.save(fh, X_year_array)

	# Saves Y_array
	with open(os.path.join(SAVE_PATH, "Y_array.npz"), "wb") as fh:
		np.save(fh, Y_array)
	
	# Saves X_word_array 
	with open(os.path.join(SAVE_PATH, "X_word_array.npz"), "wb") as fh:
	 	np.save(fh, X_word_array)

	# Saves embedding matrix 
	with open(os.path.join(SAVE_PATH, "embed_mat.npz"), "wb") as fh:
		np.save(fh, embed_mat)
