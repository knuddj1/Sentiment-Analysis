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
      

def myprint(s):
   with open('sum.txt', 'w+') as f:
       print(s, file=f)
       
        


model = Sequential()
model.add(Embedding(30001, 100, input_length=500))
model.add(Conv1D(filters=64, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(GRU(100, recurrent_dropout=0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary(print_fn=myprint)

    
