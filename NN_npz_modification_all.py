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
    years = [1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000]
    sample_size = 10000

    for year in years:
        print(year)
        filename_word = "X_word_array_" + str(year) +  "s.npz"
        filename_year = "X_year_array_" + str(year) + "s.npz"
        filename_Y = "Y_array_" + str(year) + "s.npz"
       

        X_word_array = np.load(os.path.join(X_WORD_PATH, filename_word))
        X_year_array = np.load(os.path.join(X_YEAR_PATH, filename_year))
        Y_array = np.load(os.path.join(Y_PATH, filename_Y))

        print("loaded")
        indices = np.random.choice(range(X_word_array.shape[0]), sample_size)
        X_word_array_mod = X_word_array[indices, :, :]
        X_year_array_mod = X_year_array[indices, ]
        Y_array_mod = Y_array[indices, ]
        print("indiced")

        filename_word_save = "X_word_array_" + str(year) + "s_MOD.npz"
        filename_year_save = "X_year_array_" + str(year) +  "s_MOD.npz"
        filename_Y_save = "Y_array_" + str(year) + "s_MOD.npz"

        with open(os.path.join(MOD_WORD_PATH, filename_word_save), "wb") as fh:
            np.save(fh, X_word_array_mod)
            del X_word_array
            del X_word_array_mod

        with open(os.path.join(MOD_YEAR_PATH, filename_year_save), "wb") as fh:
            np.save(fh, X_year_array_mod)
            del X_year_array
            del X_year_array_mod

        with open(os.path.join(MOD_Y_PATH, filename_Y_save), "wb") as fh:
            np.save(fh, Y_array_mod)
            del Y_array
            del Y_array_mod

