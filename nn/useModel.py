import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical



model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])


model.load_weights('model_2_64.h5')



def evaluate(array):
    testImage = np.asarray(array, dtype=uint8)
    testImage = testImage - 0.5
    testImage = testImage.reshape((-1,784))

    prediction = model.predict([testImage])
    return prediction[0]
