import numpy as np 
import tensorflow as tf 
import os
import pickle as pk
from gensim.models import KeyedVectors 
#import dircache
import re

import sklearn as sk

from sklearn.preprocessing import LabelEncoder

#Declare Paths
DATA_PATH = "/home/accts/gfs22/LING_380/Data"
EMBED_PATH = os.path.join(DATA_PATH, "Embeddings/word2vec/data")
CORPUS_PATH = os.path.join(DATA_PATH, "Corpus")
SAVE_PATH = os.path.join(DATA_PATH, "Extracted")
LEX_PATH = os.path.join(DATA_PATH, "Embeddings/lexicon.txt")

#Set Constants
MAX_THRESHOLD = 600000
NUM_VOCAB = 3954340
sent_length = 32
embed_size = 300 

ignore = {"@", "--", "-- ", " --"}
dashes = r"[- ]*"

#Function to Load Embeddings
def load_embeddings():
	print("EMBEDDING START")
	model = KeyedVectors.load_word2vec_format(EMBED_PATH, binary=True)
	print("EMBEDDING STOP")
	return model

#Function to remove dashes from words and convert words to lower case
def format_word(word):
	return re.sub(dashes, "", word.lower())

#Function to use lexicon to initialize embedding matrix, list of POS, label encoder for POS, 
#and dictionary that maps words to their embeddings
def read_lex(embeddings):

	#Open Lexicon
	with open(LEX_PATH) as fh:
		lines = fh.readlines()

	num_words = NUM_VOCAB

	#Initialize Embedding Matrix to Random Normally Distributed Numbers
	embed_mat = np.random.normal(size=(num_words, 300))
	poss = []
	words = {}

	#Generate list of all unique POS
	#Fill in embed_mat appropriately
	#Fill in words, a dictionary that maps words to actual embeddings
	for line in lines:
		word_list = line.strip().split("\t")
		if len(word_list) == 5:
			wid, _, word, _, pos = word_list[:5]
			word = format_word(word)
			wid = int(wid)
			pos = format_POS(pos)
			if word not in words:
				words[word] = wid
			poss.append(pos)
			if word in embeddings:
				embed_mat[wid, :] = embeddings[word]
		else:
			poss.append("N/A")

	#Fit Label Encoder to part of speech tags
	le = LabelEncoder()
	le.fit(poss)

	return le, poss, words, embed_mat

#Function to generate indices for word embeddings (X_word), year (X_year), and POS tags for each file
def read_in(file_name, le, updated_pos_tags, words):

	#Extract the Year
	split_fn = file_name.split("_")
	X_year = int(split_fn[-2])

	#Open file
	with open(file_name) as fh:
		lines = fh.readlines()

	#Initialize variables
	num_words = len(lines)

	sent_X_word = []
	sent_Y = []

	X_word = np.zeros(shape=(sent_length,), dtype=np.int32)
	Y = np.zeros(shape=(sent_length, ), dtype=np.int32)
	counter = 0

	#Print Name of File 
	print(file_name)
	for i, line in enumerate(lines):
		word_list = line.strip().split("\t")

		if len(word_list)==3 and word_list[0] not in ignore:

			#Format word and pos
			word, lemma, pos = word_list[:3]
			pos = format_POS(pos)
			word = format_word(word)

			#Removes words that begin with @ and words not in vocabulary
			if word.startswith("@") or word not in words:
				continue

			#Append indices for word embeddings to X_word and POS tags to Y for each word
			if counter < sent_length:
				X_word[counter] = words[word]
				Y[counter] = le.transform([pos])[0]

			#Begin a new sentence
			if lemma == '.':
				sent_X_word.append(X_word)
				X_word = np.zeros(shape=(sent_length,), dtype=np.int32)
				sent_Y.append(Y)
				Y = np.zeros(shape=(sent_length, ), dtype=np.int32)
				counter = 0

			#Iterate counter if we are not beginning a new sentence
			else:
				counter += 1


	#If we have sentences in a document join data from all sentences to make combined arrays
	if sent_X_word!=[]:
		X_word_array = np.stack(sent_X_word, axis=0)
		X_year_array = np.repeat(X_year, len(sent_X_word))
		Y_array = np.stack(sent_Y, axis=0)
		return X_word_array, X_year_array, Y_array

	#If there are no sentences in a document return all None values.
	else:
		return None, None, None

#Format POS by taking all information before the first underscore
#This gives us 423 POS
format_POS = lambda pos: pos.split("_")[0]

if __name__ == '__main__':

	#Load Embeddings
	embeddings = load_embeddings()

	#Load the Lexicon
	le, updated_pos_tags, words, embed_mat = read_lex(embeddings)

	#Create a set with all the POS tags
	updated_pos_tags = set(updated_pos_tags)

	#Save Embedding MAtrix
	with open(os.path.join(SAVE_PATH, "EMBED_MAT_FINAL_R.npz"), "wb") as fh:
		np.save(fh, embed_mat)

	#Initialize intermediate variables to store data from all files
	X_word_arrays, X_year_arrays, Y_arrays = [], [], []

	#Loop through at most 300 files from each decade
	#Append results from each file to X_word_arrays, X_year_arrays, and Y_arrays
	for dirpath, dirnames, filenames in os.walk(CORPUS_PATH):
		for file in dirnames:
			num = 0
			file_path = os.path.join(CORPUS_PATH, file)
			num_caches = len(os.listdir(file_path))
			if num_caches>300:
				filenames_chosen = np.random.choice(os.listdir(file_path), 300)
			else:
				filenames_chosen = os.listdir(file_path)
			for filename in filenames_chosen:
				print(num)
				num += 1
				X_word_array, X_year_array, Y_array = read_in(os.path.join(file_path, filename), le, updated_pos_tags, words)
				if X_word_array is not None and X_year_array is not None and Y_array is not None:
					X_word_arrays.append(X_word_array)
					X_year_arrays.append(X_year_array)
					Y_arrays.append(Y_array)

	#Concatenate result along 0 axis
	X_word_array = np.concatenate(X_word_arrays, axis=0)
	X_year_array = np.concatenate(X_year_arrays, axis=0)
	Y_array = np.concatenate(Y_arrays, axis=0)

	#Save X_word_array
	with open(os.path.join(SAVE_PATH, "X_word_array_FINAL_R.npz"), "wb") as fh:
		np.save(fh, X_word_array)

	#Save X_year_array
	with open(os.path.join(SAVE_PATH, "X_year_array_FINAL_R.npz"), "wb") as fh:
		np.save(fh, X_year_array)

	#Save Y_array
	with open(os.path.join(SAVE_PATH, "Y_array_FINAL_R.npz"), "wb") as fh:
		np.save(fh, Y_array)
	
	#Only store indices in X_word for embeddings of 600000 most common words
	indices = np.where(X_word_array>MAX_THRESHOLD)
	X_word_array[indices]=0

	#Only store embeddings of 600,000 most common words
	embed_mod = embed_mat[:MAX_THRESHOLD,]

	#Save version of X_word_array with only indices for 600,000 most common words
	with open(os.path.join(SAVE_PATH, "X_word_array_FINAL_TRIMMED_R.npz"), "wb") as fh:
	 	np.save(fh, X_word_array)

	#Save version of embedding matrix with only embeddings for 600,000 most common words
	with open(os.path.join(SAVE_PATH, "EMBED_MAT_FINAL_TRIMMED_R.npz"), "wb") as fh:
		np.save(fh, embed_mod)