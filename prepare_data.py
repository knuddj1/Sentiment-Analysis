from keras.preprocessing.sequence import pad_sequences
import csv
import numpy as np
from sklearn.utils import shuffle

def prepare_labels(arr, n_cats):
    n_samples = len(arr)
    y = np.zeros(shape=(n_samples, n_cats))
    for i in range(n_samples):
        label = int(arr[i])
        y[i, label+1] = 1.
    return y
    

def prepare(X, y, max_size, n_cats, shuffle_data=False):
    X_prepared = pad_sequences(X, maxlen=max_size, truncating='post', padding='pre')
    y_prepared = prepare_labels(y, n_cats)
    if shuffle_data:
        X_prepared, y_prepared = shuffle(X_prepared, y_prepared, random_state=0)
    return X_prepared, y_prepared