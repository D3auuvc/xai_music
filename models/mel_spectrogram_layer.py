import tensorflow as tf
from tensorflow.keras.layers import Layer


class MelSpectrogramLayer(Layer):
    def __init__(self, sr=16000, n_mels=128, n_fft=512, win_length=400, hop_length=160, pad_end=True, **kwargs):
        super(MelSpectrogramLayer, self).__init__(**kwargs)
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.pad_end = pad_end

    def call(self, input_signal):
        # Compute the Short-time Fourier Transform (STFT)
        stft = tf.signal.stft(input_signal,
                              frame_length=self.win_length,
                              frame_step=self.hop_length,
                              fft_length=self.n_fft,
                              pad_end=self.pad_end)

        # Compute magnitude of the STFT
        magnitude = tf.abs(stft)

        # Convert magnitude to decibel
        magnitude_to_db = tf.math.log(
            magnitude + 1e-6) / tf.math.log(tf.constant(10, dtype=magnitude.dtype)) * 20

        # Create a mel filterbank
        num_spectrogram_bins = stft.shape[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=self.sr,
            lower_edge_hertz=0,
            upper_edge_hertz=(self.sr / 2))

        # Apply the mel filterbank
        mel_spectrogram = tf.tensordot(
            magnitude_to_db, linear_to_mel_weight_matrix, 1)

        # Ensure the output is channels_last
        mel_spectrogram = tf.expand_dims(mel_spectrogram, -1)

        return mel_spectrogram

    def get_config(self):
        config = super(MelSpectrogramLayer, self).get_config()
        config.update({
            "sr": self.sr,
            "n_mels": self.n_mels,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "pad_end": self.pad_end
        })
        return config
