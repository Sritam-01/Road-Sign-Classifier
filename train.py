import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import tensorflow as tf

data = []
labels = []

classes = 43

# Load images
for i in range(classes):
    print(f"Loading class {i}")
    path = os.path.join('dataset/Train', str(i))

    images = os.listdir(path)

    for img in images:
        try:
            image = Image.open(path + '/' + img)
            image = image.resize((30,30))
            image = np.array(image)

            data.append(image)
            labels.append(i)

        except:
            print("Error loading image")

data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42
)

# Convert labels
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Build CNN
model = Sequential()

model.add(Conv2D(filters=32,
                 kernel_size=(5,5),
                 activation='relu',
                 input_shape=X_train.shape[1:]))

model.add(Conv2D(filters=32,
                 kernel_size=(5,5),
                 activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 activation='relu'))

model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(rate=0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(43, activation='softmax'))

# Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train
epochs = 5

model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=epochs,
    validation_data=(X_test, y_test)
)

# Save model
model.save("model/road_sign_model.h5")

print("MODEL SAVED")