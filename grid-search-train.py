import csv
import numpy as np
import json
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from gensim.parsing.preprocessing import *


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
        label = arr[i]
        if label == -1:
            y[i, 0] = 1.
        elif label == 0:
            y[i, 1] = 1.
        else:
            y[i, 2] = 1.
    return y
    


labels = []
data = []

print('collecting labels')
with open('labels.csv', 'r') as f:
    for row in csv.reader(f):
        labels.append(int(row[0]))
f.close()
print('collected labels successfully')

print('collecting data')
with open('data.csv', 'r') as f:
    for row in csv.reader(f):
        nums = [0] * len(row)
        for i, d in enumerate(row):
            nums[i] = int(d)
        data.append(nums)
f.close()
print('collected data successfully')


max_size = 500
num_categories = 3
vocab_size = 30000 + 1
embedding_size = 100
nb_epochs = 10

print('preparing data..')
x_train = pad_sequences(data, maxlen=max_size, truncating='post', padding='pre')
y_train = prepare_labels(labels, num_categories)  

del data
del labels

x_train, y_train = shuffle(x_train, y_train, random_state=0)

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
                
x_test = pad_sequences(data[1:], maxlen=59, truncating='post', padding='pre')
y_test = prepare_labels(labels[1:], num_categories)
print('data prepared.')      




model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=max_size))
model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(GRU(100, recurrent_dropout=0.3, return_sequences=True))
model.add(Conv1D(filters=16, kernel_size=4, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(GRU(100, recurrent_dropout=0.3))
model.add(Dense(250, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

es_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

model.fit(x_train, y_train, batch_size=128, shuffle=True, epochs=nb_epochs, callbacks=[es_cb], validation_split=0.2, verbose=1)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")