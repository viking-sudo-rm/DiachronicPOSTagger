# Detecting Syntactic Change Using a Neural Part-of-Speech Tagger

Code for our paper, which will appear at the 1st International Workshop on Computational Approaches to Historical Language Change  workshop at ACL 2019.

## Abstract

*We train a diachronic long short-term memory (LSTM) part-of-speech tagger on a large corpus of American English from the 19th, 20th, and 21st centuries. We analyze the tagger's ability to implicitly learn temporal structure between years, and the extent to which this knowledge can be transferred to date new sentences. The learned year embeddings show a strong linear correlation between their first principal component and time. We show that temporal information encoded in the model can be used to predict novel sentences' years of composition relatively well. Comparisons to a feedforward baseline suggest that the temporal change learned by the LSTM is syntactic rather than purely lexical. Thus, our results suggest that our tagger is implicitly learning to model syntactic change in American English over the course of the 19th, 20th, and early 21st centuries.*

<!-- TODO: Add link to paper, citations, etc. -->

## Dependencies

Our implementation uses the following Python dependencies:
* argparse
* collections
* gensim
* numpy
* os
* pickle
* random
* re
* sklearn
* matplotlib
* statsmodels
* sys
* tensorflow

All of these libraries can be installed with pip.

## Getting Started

Please contact the authors for data.

Once you have the raw data downloaded, the data processing file data_processing.py must be run first. Please specify `EMBED_PATH` (the location of the word embeddings -- do not include the name of the embedding file), `CORPUS PATH` (the location of the text files -- do not include the name of any text file), `SAVE_PATH` (the location you would like to save the output embedding matrix, `X_word_array`, `X_year_array`, and `Y_array`), and `LEX_PATH` (the location of the lexicon file -- include the lexicon filename).

The actual code to train and evaluate the LSTM (lstm.py) must be run second. We must specify the `DATA_PATH` (the location of the processed `X_word_array`, `X_year_array`, and `Y_year_array` -- do not include any of the array filenames), `LEX_PATH` (the location of the lexicon file including the lexicon filename), `TRAIN_SAVE_PATH`/`TEST_SAVE_PATH` (the location to save the train and test data, respectively), `MODEL_PATH` (the location to save all model information), and `PLOTS_PATH` (the location to save all plots).

`EMBED_PATH` (the location of the embedding matrix including the name of the embedding matrix file), `TRAIN_SAVE_PATH`/`DEV_SAVE_PATH`/`TEST_SAVE_PATH` (the location to save the train, dev and test data respectively), `WRITE_TO_PATH` (the location to which the output of the train function will be written), and `X_WORD_FILENAME`/`X_YEAR_FILENAME`/`Y_FILENAME` (the filenames of the processed `X_word_array`, `X_year_array`, and `Y_array`).

After the data has been downloaded and is located correctly, the data processing file can be run from the terminal using the command:

```
python dataprocessing.py
```

After the data is processed, you can train an LSTM model and test it using:

```
python lstm.py --cut
```

The `--cut` flag specifies that you want to create a test, train, and dev set. Once you have run the command with this flag once, you can leave it out in the future to use the same dataset.

In addition, once you have a trained model, you can rerun the evaluation code without retraining by running:

```
python lstm.py --notrain
```

Additional `argparse` options can be found in the file.

## License

We obtained the rights to use the Corpus of Historical American English (COHA) through our affiliation with Yale University. Thank you Kevin Merriman for helping us get access to this corpus!


