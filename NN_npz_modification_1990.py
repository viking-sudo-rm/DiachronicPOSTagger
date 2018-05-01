import os
import numpy as np

DATA_PATH = "/home/accts/gfs22/LING_380/Data"
SAVE_PATH = os.path.join(DATA_PATH, "Extracted")
X_WORD_PATH = os.path.join(SAVE_PATH, "X_word")
X_YEAR_PATH = os.path.join(SAVE_PATH, "X_year")
Y_PATH = os.path.join(SAVE_PATH, "Y")
MOD_WORD_PATH = os.path.join(SAVE_PATH, "X_word_10000")
MOD_YEAR_PATH = os.path.join(SAVE_PATH, "X_year_10000")
MOD_Y_PATH = os.path.join(SAVE_PATH, "Y_10000")

if __name__ == "__main__":

    #  TODO #
    filename_word = "X_word_array_1860s.npz"
    filename_year = "X_year_array_1860s.npz"
    filename_Y = "Y_array_1860s.npz"
    sample_size = 10000

    X_word_array = np.load(os.path.join(X_WORD_PATH, filename_word))
    print("loaded word")
    X_year_array = np.load(os.path.join(X_YEAR_PATH, filename_year))
    Y_array = np.load(os.path.join(Y_PATH, filename_Y))

    print("loaded")
    indices = np.random.choice(range(X_word_array.shape[0]), sample_size)
    X_word_array_mod = X_word_array[indices, :, :]
    X_year_array_mod = X_year_array[indices, ]
    Y_array_mod = Y_array[indices, ]
    print("indexed")

    with open(os.path.join(MOD_WORD_PATH, "X_word_array_1860s_MOD.npz"), "wb") as fh:
        np.save(fh, X_word_array_mod)


    with open(os.path.join(MOD_YEAR_PATH, "X_year_array_1860s_MOD.npz"), "wb") as fh:
        np.save(fh, X_year_array_mod)

    with open(os.path.join(MOD_Y_PATH, "Y_array_1860s_MOD.npz"), "wb") as fh:
        np.save(fh, Y_array_mod)
        

