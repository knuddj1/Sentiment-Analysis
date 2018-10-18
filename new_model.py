import os
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
from keras.layers import CuDNNGRU
from keras.layers import CuDNNLSTM
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import AveragePooling1D
from keras.layers import SpatialDropout1D   
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


MAX_SIZE = 200
NUM_CATEGORIES = 3

print('preparing data..')
x_train = pad_sequences(data, maxlen=MAX_SIZE, truncating='post', padding='pre')
y_train = prepare_labels(labels, NUM_CATEGORIES)  

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
                
x_test = pad_sequences(data[1:], maxlen=MAX_SIZE, truncating='post', padding='pre')
y_test = prepare_labels(labels[1:], NUM_CATEGORIES)

del data
del labels
del nums
del d
del words
del reader

print('data prepared.')     

EMBEDDING_SIZE = 200
N_RECURRENT_UNITS = 500
DROPOUT = 0.3
VOCAB_SIZE = 30000 + 1
NB_EPOCHS = 10

base_dir = 'experiment_model {0} Epochs/'.format(NB_EPOCHS)
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE, input_length=MAX_SIZE))
model.add(Dropout(DROPOUT))
#model.add(LSTM(N_RECURRENT_UNITS, recurrent_dropout=DROPOUT, return_sequences=True))
#model.add(LSTM(N_RECURRENT_UNITS, recurrent_dropout=DROPOUT))
model.add(Conv1D(filters=512, kernel_size=5, padding='same'))
model.add(AveragePooling1D(pool_size=5))
model.add(Flatten())
model.add(Dense(250, activation='tanh'))
model.add(Dense(250, activation='tanh'))
model.add(Dense(NUM_CATEGORIES, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
history = model.fit(x_train, y_train, batch_size=128, shuffle=True, epochs=NB_EPOCHS, callbacks=[es], validation_split=0.2, verbose=1)

acc =       history.history['acc']
val_acc =   history.history['val_acc']
loss =      history.history['loss']
val_loss =  history.history['val_loss']

test_loss, test_acc = model.evaluate(x_test,y_test)


os.chdir(base_dir)
results_dic = {
    'acc':       acc,
    'loss':      loss,
    'val_acc':   val_acc,
    'val_loss':  val_loss,
    'test_acc':  test_acc,
    'test_loss': test_loss
}
with open('results.json', 'w') as f:
    json.dump(results_dic, f)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
         