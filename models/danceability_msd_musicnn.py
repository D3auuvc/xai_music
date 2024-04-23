import tensorflow as tf
from keras import layers, models

class DanceabilityMSDMusicnn:
    def __init__(self, pre_trained_model_path, n_classes, sample_rate, delta_time):
        self.pre_trained_model_path = pre_trained_model_path
        self.n_classes = n_classes
        self.sample_rate = sample_rate
        self.delta_time = delta_time
        self.model = self.build_model()

    def load_pretrained_model(self):
        # Attempt to load the pre-trained model directly with TensorFlow SavedModel loader for better compatibility
        return tf.keras.models.load_model(self.pre_trained_model_path, compile=False)

    def build_model(self):
        input_length = 200  # As specified in the schema.inputs.shape

        # Load and wrap the pre-trained model
        pre_trained_model = self.load_pretrained_model()

        # Create a new model suitable for audio classification, compatible with TensorFlow 2.4.0
        new_model = models.Sequential([
            pre_trained_model,
            layers.Reshape((input_length, 1)),  # Adjusted as necessary
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.n_classes, activation='softmax')
        ])

        # Compile the model
        new_model.compile(optimizer='adam',  # You can choose another optimizer
                          loss='categorical_crossentropy',  # Or another loss function suitable for your problem
                          metrics=['accuracy'])  # You can add more metrics if needed

        return new_model

