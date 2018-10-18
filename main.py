from model import Model
from test_data import get_test_data
from train_data import get_train_data
from keras.callbacks import EarlyStopping

MAX_SIZE = 500
NUM_CATEGORIES = 3
NB_EPOCHS = 10

# Retrieving data
x_train, y_train = get_train_data(max_size=MAX_SIZE, n_cats=NUM_CATEGORIES)
x_test, y_test = get_test_data(max_size=MAX_SIZE, n_cats=NUM_CATEGORIES)


#training, testing and saving model
model = Model(max_size=MAX_SIZE, n_cats=NUM_CATEGORIES)
model.train(x_train=x_train, y_train=y_train, nb_epochs=NB_EPOCHS)
model.test(x_test=x_test, y_test=y_test)
model.save()