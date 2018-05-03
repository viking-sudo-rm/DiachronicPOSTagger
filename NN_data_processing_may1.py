import numpy as np 
import tensorflow as tf 
import fasttext, os
sent_length = 32
embed_size = 300 #given from fasttext website
import pickle as pk
from gensim.models import KeyedVectors 
import dircache

import sklearn as sk

from sklearn.preprocessing import LabelEncoder

DATA_PATH = "/home/accts/gfs22/LING_380/Data"
TAGS_PATH = os.path.join(DATA_PATH, "Tags/pos.pkl")
EMBED_PATH = os.path.join(DATA_PATH, "Embeddings/word2vec/data")
CORPUS_PATH = os.path.join(DATA_PATH, "Corpus")
SAVE_PATH = os.path.join(DATA_PATH, "Extracted")
LEX_PATH = os.path.join(DATA_PATH, "Embeddings/lexicon.txt")

MAX_THRESHOLD = 600000

def load_embeddings():
	print("EMBEDDING START")
	model = KeyedVectors.load_word2vec_format(EMBED_PATH, binary=True)
	#model = fasttext.load_model(EMBED_PATH)
	print("EMBEDDING STOP")
	return model

def read_lex(embeddings):

	with open(LEX_PATH) as fh:
		lines = fh.readlines()

	num_words = len(lines) + 1

	embed_mat = np.ones([num_words, 300])
	poss = []
	words = {}

	for line in lines:
		word_list = line.strip().split("\t")
		if len(word_list) == 5:
			wid, _, word, _, pos = word_list[:5]
			if "--" in word:
				continue
				# print "--", word_list
			wid = int(wid)
			pos = format_POS(pos)
			words[word] = wid
			poss.append(pos)
			if word in embeddings:
				embed_mat[wid, :] = embeddings[word]
		else:
			poss.append("N/A")

	le = LabelEncoder()
	le.fit(poss)

	return le, poss, words, embed_mat


#	print("ENCODE POS")
#	pos_tags = pk.load(open(TAGS_PATH))
#	updated_pos_tags = [format_POS(tag) for tag in pos_tags]
#	le = LabelEncoder()
#	le.fit(updated_pos_tags)
#
#	return le, updated_pos_tags


def read_in(file_name, embeddings, le, updated_pos_tags, words):

	print("READ IN  START")

	#Extract the Year
	split_fn = file_name.split("_")

	X_year = int(split_fn[-2])

	with open(file_name) as fh:
		lines = fh.readlines()

	num_words = len(lines)

	sent_X_word = []
	sent_Y = []

	X_word = np.zeros(shape=(sent_length,), dtype=np.int32)
	Y = np.zeros(shape=(sent_length, ), dtype=np.int32)
	counter = 0
	print("READ IN  LINES")
	print(filename)
	for line in lines:
		word_list = line.split()
		#print(word_list)
		if len(word_list)==3 and word_list[0]!="@" and counter<sent_length:
			word, lemma, pos = word_list[:3]
			pos = format_POS(pos)
			if pos in updated_pos_tags:
				word = word.lower()
				if word in embeddings:
					X_word[counter] = words[word]
				else:
					X_word[counter] = 0
				Y[counter] = le.transform([pos])[0]
				if lemma == '.' or counter==sent_length-1:
					#print("BET")
					#print(counter)
					sent_X_word.append(X_word)
					X_word = np.zeros(shape=(sent_length,), dtype=np.int32)
					sent_Y.append(Y)
					Y = np.zeros(shape=(sent_length, ), dtype=np.int32)
					counter = 0
				else:
					#print(counter)
					counter += 1

	#print(sent_X_word)

	if sent_X_word!=[]:
		X_word_array = np.stack(sent_X_word, axis=0)

		X_year_array = np.repeat(X_year, len(sent_X_word))

		Y_array = np.stack(sent_Y, axis=0)
		print("READ IN  END")

		return X_word_array, X_year_array, Y_array
	else:
		return None, None, None

format_POS = lambda pos: pos.split("_")[0]

if __name__ == '__main__':

	embeddings = load_embeddings()
	le, updated_pos_tags, words, embed_mat = read_lex(embeddings)
	#embed_mod = embed_mat[:100,]

	#print le.transform(updated_pos_tags[:100])
	updated_pos_tags = set(updated_pos_tags)

	
	#with open(os.path.join(SAVE_PATH, "EMBED_MAT.npz"), "wb") as fh:
	#	np.save(fh, embed_mat)

	#with open(os.path.join(SAVE_PATH, "EMBED_MAT_MOD.npz"), "wb") as fh:
	#	np.save(fh, embed_mod)

	X_word_arrays, X_year_arrays, Y_arrays = [], [], []

	
	for dirpath, dirnames, filenames in os.walk(CORPUS_PATH):
		print dirnames
		num = 0
		for file in dirnames:
			file_path = os.path.join(CORPUS_PATH, file)
			num_caches = len(dircache.listdir(file_path))
			print file_path
			print num_caches
			if num_caches>300:
				filenames_chosen = np.random.choice(dircache.listdir(file_path), 300)
			else:
				filenames_chosen = dircache.listdir(file_path)
			#print filenames_chosen
			#print len(filenames_chosen)
			for filename in filenames_chosen:
				print(num)
				num += 1
				X_word_array, X_year_array, Y_array = read_in(os.path.join(file_path, filename), embeddings, le, updated_pos_tags, words)
				if X_word_array is not None and X_year_array is not None and Y_array is not None:
					X_word_arrays.append(X_word_array)
					X_year_arrays.append(X_year_array)
					Y_arrays.append(Y_array)


	X_word_array = np.concatenate(X_word_arrays, axis=0)
	X_year_array = np.concatenate(X_year_arrays, axis=0)
	Y_array = np.concatenate(Y_arrays, axis=0)

	#with open(os.path.join(SAVE_PATH, "X_word_array_5_1.npz"), "wb") as fh:
	#	np.save(fh, X_word_array)

	with open(os.path.join(SAVE_PATH, "EMBED_MAT.npz"), "wb") as fh:
	 	np.save(fh, embed_mat)

	with open(os.path.join(SAVE_PATH, "X_word_array_5_2.npz"), "wb") as fh:
		np.save(fh, X_word_array)

	indices = np.where(X_word_array>MAX_THRESHOLD)
	X_word_array[indices]=0
	embed_mod = embed_mat[:MAX_THRESHOLD,]

	with open(os.path.join(SAVE_PATH, "EMBED_MOD.npz"), "wb") as fh:
	 	np.save(fh, embed_mod)


	# with open(os.path.join(SAVE_PATH, "EMBED_MAT_MOD.npz"), "wb") as fh:
	# 	np.save(fh, embed_mod)

	with open(os.path.join(SAVE_PATH, "X_word_array_600000.npz"), "wb") as fh:
		np.save(fh, X_word_array)

	#with open(SAVE_PATH) as fh:
#		np.save(os.path.join(fh, "X_word_array_1810s.npz"), X_word_array)

#	with open(SAVE_PATH) as fh:
#		np.save(os.path.join(fh, "X_year_array_1810s.npz"), X_year_array)

#	with open(SAVE_PATH) as fh:
#		np.save(os.path.join(fh, "Y_array_1810s.npz"), Y_array)

#	with open(os.path.join(SAVE_PATH, "X_word_array_1810s.npz"), "wb") as fh:
#    	np.save(fh, X_word_array)

#    with open(os.path.join(SAVE_PATH, "X_year_array_1810s.npz"), "wb") as fh:
#    	np.save(fh, X_year_array)

#    with open(os.path.join(SAVE_PATH, "Y_array_1810s.npz"), "wb") as fh:
#    	np.save(fh, Y_array)

	with open(os.path.join(SAVE_PATH, "X_year_array_5_2.npz"), "wb") as fh:
		np.save(fh, X_year_array)

	with open(os.path.join(SAVE_PATH, "Y_array_5_2.npz"), "wb") as fh:
		np.save(fh, Y_array)
	
	
	#with open(os.path.join(SAVE_PATH, "X_test.npz"), "wb") as fh:
    #	np.save(fh, X)

#    with open(SAVE_PATH, "wb") as fh:
#		np.save(os.path.join(fh, "X_test.npz"), X)

 #   np.save(open(os.path.join(SAVE_PATH, "X_test.npz")), X)

