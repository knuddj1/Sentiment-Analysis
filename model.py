import json
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
from keras.layers import GlobalMaxPooling1D
from keras.layers import AveragePooling1D
from keras.layers import SpatialDropout1D   
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping

EMBEDDING_SIZE = 200
N_RECURRENT_UNITS = 100
DROPOUT = 0.3
VOCAB_SIZE = 30000 + 1

class Model:
    def __init__(self, max_size, n_cats):
        self.model = self.create_model(max_size=max_size, n_cats=n_cats)
        self.train_results = None
        self.test_results = None

    def create_model(self, max_size, n_cats):
        print("creating model..")
        model = Sequential()
        model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE, input_length=max_size))
        model.add(Conv1D(filters=64, kernel_size=8, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(N_RECURRENT_UNITS, recurrent_dropout=DROPOUT, return_sequences=True))
        model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(N_RECURRENT_UNITS, recurrent_dropout=DROPOUT))
        model.add(Dense(250, activation='tanh'))
        model.add(Dense(n_cats, activation='softmax'))
        print("created model successfully.")
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, nb_epochs):
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto')
        self.train_results = self.model.fit(x_train, y_train, batch_size=128, shuffle=True, epochs=nb_epochs,  callbacks=[es], validation_split=0.2, verbose=1).history

    def test(self, x_test, y_test):
        self.test_results = self.model.evaluate(x_test,y_test)

    def save(self):
        results_dic = {
            'acc':       self.train_results['acc'],
            'loss':      self.train_results['loss'],
            'val_acc':   self.train_results['val_acc'],
            'val_loss':  self.train_results['val_loss'],
            'test_acc':  self.test_results[0],
            'test_loss': self.test_results[1]
        }
        with open('results.json', 'w') as f:
            json.dump(results_dic, f)

        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("model.h5")
        print("Saved model to disk")

    def load_model(self):
        print("loading model..")
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("loaded model.")
        loaded_model.summary()
        self.model = loaded_model
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
