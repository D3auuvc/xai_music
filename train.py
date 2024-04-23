import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import os
from scipy.io import wavfile
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models import Conv1D
from tqdm import tqdm
from glob import glob
import argparse
import warnings


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # Initialize X and Y after computing the first spectrogram to get its shape
        first_wav_path = wav_paths[0]
        rate, first_wav = wavfile.read(first_wav_path)
        first_spectrogram = self.get_spectrogram(first_wav)
        spectrogram_shape = first_spectrogram.shape
        X = np.empty((self.batch_size, *spectrogram_shape), dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        # Now that X is properly initialized, fill it with spectrograms
        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            spectrogram = self.get_spectrogram(wav)
            if spectrogram.shape != spectrogram_shape:
                raise ValueError(
                    f"Spectrogram shape mismatch: expected {spectrogram_shape}, got {spectrogram.shape}")
            X[i,] = spectrogram
            Y[i,] = to_categorical(label, num_classes=self.n_classes)

        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_spectrogram(self, waveform):
        # Convert waveform from int16 to float32
        waveform = tf.cast(waveform, tf.float32)
        # Normalize the waveform between -1.0 and 1.0
        waveform = waveform / tf.int16.max

        # Convert the waveform to a spectrogram via a STFT.
        spectrogram = tf.signal.stft(
            waveform, frame_length=255, frame_step=128)
        # Obtain the magnitude of the STFT.
        spectrogram = tf.abs(spectrogram)
        # Add a `channels` dimension, so that the spectrogram can be used
        # as image-like input data with convolution layers (which expect
        # shape (`batch_size`, `height`, `width`, `channels`).
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram


def train(args):
    src_root = args.src_root
    sr = args.sample_rate
    dt = args.delta_time
    batch_size = args.batch_size
    model_type = args.model_type
    params = {'N_CLASSES': len(os.listdir(args.src_root)),
              'SR': sr,
              'DT': dt}

    # models = {'conv1d': Conv1D(**params)}
    # assert model_type in models.keys(), '{} not an available model'.format(model_type)

    # Ensure the logs directory exists
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    csv_path = os.path.join(logs_dir, '{}_history.csv'.format(model_type))

    # Get wav files
    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]

    # Get classes
    classes = sorted(os.listdir(args.src_root))
    le = LabelEncoder()
    le.fit(classes)
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    labels = le.transform(labels)

    # Split data
    wav_train, wav_val, label_train, label_val = train_test_split(wav_paths,
                                                                  labels,
                                                                  test_size=0.1,
                                                                  random_state=0)

    assert len(
        label_train) >= args.batch_size, 'Number of train samples must be >= batch_size'
    if len(set(label_train)) != params['N_CLASSES']:
        warnings.warn('Found {}/{} classes in training data. Increase data size or change random_state.'.format(
            len(set(label_train)), params['N_CLASSES']))
    if len(set(label_val)) != params['N_CLASSES']:
        warnings.warn('Found {}/{} classes in validation data. Increase data size or change random_state.'.format(
            len(set(label_val)), params['N_CLASSES']))

    tg = DataGenerator(wav_train, label_train, sr, dt,
                       params['N_CLASSES'], batch_size=batch_size)
    vg = DataGenerator(wav_val, label_val, sr, dt,
                       params['N_CLASSES'], batch_size=batch_size)

    print(tg[0][0].shape)  # This will print the shape of X
    print(tg[0][1].shape)  # This will print the shape of Y

    # model = models[model_type]
    # cp = ModelCheckpoint('models/{}.keras'.format(model_type), monitor='val_loss',
    #                      save_best_only=True, save_weights_only=False,
    #                      mode='auto', save_freq='epoch', verbose=1)
    # csv_logger = CSVLogger(csv_path, append=False)
    # model.fit(tg, validation_data=vg,
    #           epochs=30, verbose=1,
    #           callbacks=[csv_logger, cp])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Audio Classification Training')
    parser.add_argument('--model_type', type=str, default='conv1d',
                        help='model to run. i.e. conv1d')
    parser.add_argument('--src_root', type=str, default='clean',
                        help='directory of audio files in total duration')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sample_rate', '-sr', type=int, default=16000,
                        help='sample rate of clean audio')
    args, _ = parser.parse_known_args()

    train(args)
