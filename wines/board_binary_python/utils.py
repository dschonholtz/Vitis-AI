import os
import numpy as np


def get_input_data(input_dir):
    """
    Takes a dir of csvs, pulls out all of the csv files and puts them into a 2d numpy array.
    """
    input_data = []
    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            input_file = os.path.join(input_dir, file)
            input_chunk = np.loadtxt(input_file, delimiter=',', dtype='float32')
            input_data.append(input_chunk)
    return input_data