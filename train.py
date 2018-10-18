import csv
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import LSTM 
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle

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


num_samples = len(labels)
max_size = 500
num_categories = 3
vocab_size = 30000 + 1
embedding_size = 100
nb_epochs = 10

print('preparing data..')
x_train = pad_sequences(data, maxlen=max_size, truncating='post', padding='pre')
y_train = np.zeros(shape=(num_samples, num_categories))

for i in range(num_samples):
    label = labels[i]
    if label == -1:
        y_train[i, 0] = 1.
    elif label == 0:
        y_train[i, 1] = 1.
    else:
        y_train[i, 2] = 1.     
print('data prepared.')
del data
del labels


x_train, y_train = shuffle(x_train, y_train, random_state=0)

model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=max_size))
model.add(Conv1D(filters=64, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, recurrent_dropout=0.5, return_sequences=True))
model.add(Conv1D(filters=32, kernel_size=4, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, recurrent_dropout=0.5, return_sequences=True))
model.add(Conv1D(filters=16, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(50, activation='relu'))
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