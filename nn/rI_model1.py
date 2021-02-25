#makes life easier by using the mnist data set
import numpy as np
import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5



# relying on a 2 hidden layor model with 64 neurons each
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'],)

# Train the model.
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=10,
  batch_size=32,
)

# Evaluate the model.
model.evaluate(
  test_images,
  to_categorical(test_labels)
)

# Save the model to disk.
model.save_weights('model_2_64.h5')

# Load the model from disk later using:
# model.load_weights('model.h5')

# Predict on the first 5 test images.
#predictions = model.predict(test_images[:5])

# Print our model's predictions.
#print(predictions) # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
#print(test_labels[:5]) # [7, 2, 1, 0, 4]
