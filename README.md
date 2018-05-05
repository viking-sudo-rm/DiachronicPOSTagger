# Analyzing Syntactic Change in American English Using a Time-Parametrized Long Short-Term Memory Part-of-Speech Tagger

PASTE IN ABSTRACT

### Prerequisites

We used the belowPython libraries:
1) numpy
2) tensorflow
3) gensim
4) dircache
5) sklearn

All of these libraries can be installed with pip.

## Getting Started

Please contact the authors for data.

The data processing file `data_processing.py` must be run first. Please specify EMBED_PATH (the location of the word embeddings not including the names of the embedding file), CORPUS PATH (the location of the text files not including the name of any text file), SAVE_PATH (the location you would like to save the output embedding matrix, X_word_array, X_year_array, and Y_array), and LEX_PATH (the location of the lexicon file including the lexicon file name).

The actual code to train and evaluate the LSTM `lstm.py`must be run second. We must specify the DATA_PATH (the location of the processed X_word_array, X_year_array, and Y_year_array not including the names of any array), EMBED_PATH (the location of the embedding matrix including the name of the embedding matrix), TRAIN_SAVE_PATH/DEV_SAVE_PATH/TEST_SAVE_PATH (the location to save the train, dev and test data respectively), WRITE_TO_PATH(the location to which the output of the train function will be written), and X_WORD_FILENAME/X_YEAR_FILENAME/Y_FILENAME (the filenames of the processed X_word_array, X_year_array, and Y_array.


After the data has been downloaded and is located correctly, the data processing file can be run from the terminal using the command:

```python dataprocessing.py```

After the data is processed and is located correctly, the LSTM file can be run from the terminal. 

If you would like to 

```python lstm.py```

 
## Authors

Gigi Stark and Will Merrill

## License

Yale University gave us the license to use the Corpus of Historical American English (COHA). Thank you Kevin Merriman for your help in obtaining usage rights for this corpus!


