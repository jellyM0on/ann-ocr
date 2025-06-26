# File processing functions here

import pandas as pd
import numpy as np

def load_training_data(file_path):
    df = pd.read_excel(file_path)
    X = df.iloc[:, 1:36].values.astype(int)
    y_ascii = df['label'].values

    ascii_classes = sorted(set(y_ascii))
    ascii_to_index = {val: i for i, val in enumerate(ascii_classes)}
    index_to_ascii = {i: val for val, i in ascii_to_index.items()}

    Y = np.zeros((len(y_ascii), len(ascii_classes)))
    for i, val in enumerate(y_ascii):
        Y[i, ascii_to_index[val]] = 1

    return X, Y, ascii_to_index, index_to_ascii

def extract_input_from_excel(file_path):
    df = pd.read_excel(file_path, header=None)
    df = df.reindex(index=range(7), columns=range(5), fill_value=0)
    matrix = (df.fillna(0).to_numpy() != 0).astype(int)
    return matrix.flatten(), matrix