from tensorflow import keras

model = keras.models.load_model('../model/deeparg2.h5')
ynew = model.predict(test_dataset)
