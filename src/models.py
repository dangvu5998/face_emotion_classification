import tensorflow as tf
from tensorflow.keras import layers
def simple_cnn(input_shape, num_classes, dropout=0.25):
    model = tf.keras.models.Sequential()

    model.add(layers.Conv2D(32, (7, 7), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D(padding='same'))
    model.add(layers.Dropout(dropout))

    model.add(layers.Conv2D(64, (5, 5), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D(padding='same'))
    model.add(layers.Dropout(dropout))

    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D(padding='same'))
    model.add(layers.Dropout(dropout))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model
