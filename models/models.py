import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from kapre.composed import get_melspectrogram_layer


def Conv1D(N_CLASSES=10, SR=16000, DT=1.0):
    # Input shape (None, SR * DT) where SR is sample rate, DT is duration in seconds
    input_shape = (int(SR * DT), 1)  # Mono channel

    model = Sequential()
    model.add(layers.Input(shape=input_shape))
    # # MelSpectrogram layer
    # model.add(layers.MelSpectrogram(sampling_rate=SR,
    #           num_mel_bins=128, name='mel_spectrogram'))
    model.add(layers.LayerNormalization(axis=2, name='batch_norm'))
    # model.add(layers.Reshape((-1, 128, 1), name='reshape'))  # Reshape for CNN

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
              padding='same', name='conv1'))
    model.add(layers.MaxPooling2D((2, 2), name='max_pool1'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
              padding='same', name='conv2'))
    model.add(layers.MaxPooling2D((2, 2), name='max_pool2'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu',
              padding='same', name='conv3'))
    model.add(layers.MaxPooling2D((2, 2), name='max_pool3'))

    # Classifier
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(64, activation='relu', name='dense1'))
    model.add(layers.Dropout(0.5, name='dropout'))
    model.add(layers.Dense(N_CLASSES, activation='softmax', name='output'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model
