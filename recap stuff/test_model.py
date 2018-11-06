from keras.models import model_from_json

def test(m_path, w_path, x_test, y_test):
    model_path = m_path
    weights_path = w_path

    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)
    #evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    loaded_model.summary()
    test_loss, test_acc = loaded_model.evaluate(x_test,y_test)
    print('test_loss: {} test acc: {}'.format(test_loss, test_acc))