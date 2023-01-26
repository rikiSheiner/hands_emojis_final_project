import numpy as np
import os
import cv2
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout

"""
This module is used for building hand gesture classification model based on the emoji dataset
Written by : Rivka Sheiner
"""


DATADIR = "emoji_data/data"
CATEGORIES = ['class_1', 'class_2', 'class_3', 'class_4', 'class_5']
IMG_SIZE = 100


def create_data(data_dir):
    data = []
    for category in CATEGORIES:
        path = os.path.join(data_dir, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([new_array, class_num])
            except Exception as e:
                pass

    return data


def prepare_data(data):
    x = []
    y = []

    for features, label in data:
        x.append(features)
        y.append(label)

    x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)

    return x, y


def build_model(x):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=x.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))

    num_classes = len(CATEGORIES)
    model.add(Dense(num_classes))
    model.add(Activation("sigmoid"))

    return model


def main():
    # Read and prepare the training data
    train_data = create_data(os.path.join(DATADIR, 'Train'))
    x_train, y_train = prepare_data(train_data)

    # Read and prepare the validation data
    valid_data = create_data(os.path.join(DATADIR, 'Valid'))
    x_valid, y_valid = prepare_data(valid_data)

    # Build the model
    model = build_model(x_train)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_valid, y_valid), shuffle=True)

    # Save the model
    model.save('hand_emoji_model')


if __name__ == '__main__':
    main()



