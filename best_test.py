# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:05:54 2018

@author: OP-VR
"""

import os
import csv
import numpy as np
import json
from keras.preprocessing.sequence import pad_sequences
from gensim.parsing.preprocessing import *
from keras.models import model_from_json

CUSTOM_FILTERS = [  lambda x: x.lower(), #To lowercase
                    lambda text: re.sub(r'https?:\/\/.*\s', '', text, flags=re.MULTILINE), #To Strip away URLs
                    strip_tags, #Remove tags from s using RE_TAGS.
                    strip_non_alphanum,#Remove non-alphabetic characters from s using RE_NONALPHA.
                    strip_punctuation, #Replace punctuation characters with spaces in s using RE_PUNCT.
                    strip_numeric, #Remove digits from s using RE_NUMERIC.
                    strip_multiple_whitespaces,#Remove repeating whitespace characters (spaces, tabs, line breaks) from s and turns tabs & line breaks into spaces using RE_WHITESPACE.
                    remove_stopwords, # Set of 339 stopwords from Stone, Denis, Kwantes (2010).
                    lambda x: strip_short(x, minsize=3), #Remove words with length lesser than minsize from s.
                ]

def prepare_labels(arr, n_cats):
    n_samples = len(arr)
    y = np.zeros(shape=(n_samples, n_cats))
    for i in range(n_samples):
        label = int(arr[i])
        if label == -1:
            y[i, 0] = 1.
        elif label == 0:
            y[i, 1] = 1.
        else:
            y[i, 2] = 1.
    return y

MAX_SIZE = 500
NUM_CATEGORIES = 3

with open('word_to_index_top_30000.json', 'r') as f:
    d = json.load(f)

data = []
labels = []

with open('test_data.csv', 'r', encoding="utf8") as csvfile:
    reader = csv.reader(csvfile)
    for r in reader:
        words = preprocess_string(r[0], CUSTOM_FILTERS)
        nums = [0] * len(words)
        for i, word in enumerate(words):
            if word in d:
                nums[i] = d[word]
        data.append(nums)
        labels.append(r[-1])
                
x_test = pad_sequences(data[1:], maxlen=MAX_SIZE, truncating='post', padding='pre')
y_test = prepare_labels(labels[1:], NUM_CATEGORIES)

model_path = 'models 10 epochs\Embedding size 200 - LSTM Layers 2\model.json'
weights_path = 'models 10 epochs\Embedding size 200 - LSTM Layers 2\model.h5'

json_file = open(model_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(weights_path)
#evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
test_loss, test_acc = loaded_model.evaluate(x_test,y_test)

loaded_model.summary()
print('test_loss: {} test acc: {}'.format(test_loss, test_acc))
         