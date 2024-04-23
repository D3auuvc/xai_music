import os
from tqdm import tqdm
import soundfile as sf
import librosa
from pathlib import Path
import muda
import yaml


class Audiobank:

    def __init__(self, config_file: str):
        self.config = self.__read_config_file(config_file)
        self.deformers = []

    def __read_config_file(self, config_file: str) -> dict:
        """
        Reads the configuration file and returns a dictionary containing the configuration data.
        Args:
            config_file (str): The path to the configuration file.
        Returns:
            dict: A dictionary containing the configuration data.
        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"The configuration file {config_file} does not exist."
            )
        with open(config_file, "r") as file:
            return yaml.safe_load(file)

    def __loudness(self, audio_path: str, output_path: str, gain_db: float) -> None:
        """
        Modifies the loudness of an audio file.

        Args:
            audio_path (str): Path to the input audio file.
            output_path (str): Path where the modified audio will be saved.
            gain_db (float): The gain in decibels to apply to the audio signal.
        """
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)

        # Apply gain
        y_modified = librosa.effects.percussive(y) * 10 ** (gain_db / 20.0)

        # Save the modified audio
        sf.write(output_path, y_modified, sr)

    def __get_audio_paths(self, audio_format: str = ".wav") -> list:
        """
        Retrieves a list of all .wav file paths from the directory specified in the configuration file under 'mir_dataset_path'.

        Returns:
            list: A list containing the paths of all .wav files in the specified directory.
        """
        audio_paths = []
        dataset_path = Path(self.config["mir_dataset_path"])
        for audio_file in dataset_path.rglob(f"*{audio_format}"):
            audio_paths.append(str(audio_file))

        if len(audio_paths) == 0:
            raise FileNotFoundError("No audio files found in the specified directory.")
        return audio_paths

    def __hpss(
        self, audio_path: str, output_path_harmonic: str, output_path_percussive: str
    ) -> None:
        """
        Applies Harmonic/Percussive Source Separation (HPSS) to an audio file and saves the harmonic and percussive components separately.

        Args:
            audio_path (str): Path to the input audio file.
            output_path_harmonic (str): Path where the harmonic component will be saved.
            output_path_percussive (str): Path where the percussive component will be saved.
        """
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)

        # Apply HPSS
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Save the harmonic and percussive audio
        sf.write(output_path_harmonic, y_harmonic, sr)
        # sf.write(output_path_percussive, y_percussive, sr)

    def synthesis(
        self, input_format: str = ".wav", output_format: str = ".wav"
    ) -> None:
        """
        Performs audio synthesis by applying various audio deformations specified in the configuration file.
        This method processes all audio files in the input format, applies the deformations, and saves the augmented audio and metadata in the output format.

        Args:
            input_format (str, optional): The format of the input audio files. Defaults to ".wav".
            output_format (str, optional): The format for saving the augmented audio files. Defaults to ".wav".
        """
        if "hpss" in self.config:
            if self.config["hpss"]["apply"]:
                self.__hpss(
                    self.config["hpss"]["audio_path"],
                    self.config["hpss"]["output_path_harmonic"],
                    self.config["hpss"]["output_path_percussive"],
                )
        if "tempo_factor" in self.config:
            self.deformers.append(
                (
                    "time_stretch",
                    muda.deformers.LogspaceTimeStretch(**self.config["tempo_factor"]),
                )
            )
        if "keys" in self.config:
            self.deformers.append(
                (
                    "pitch_shift",
                    muda.deformers.LinearPitchShift(**self.config["keys"]),
                )
            )
        if "drc" in self.config:
            self.deformers.append(
                (
                    "drc",
                    muda.deformers.DynamicRangeCompression(preset=self.config["drc"]),
                )
            )

        pipeline = muda.Pipeline(steps=self.deformers)
        audio_list = self.__get_audio_paths(audio_format=input_format)

        for audio_path in tqdm(audio_list):
            j_orig = muda.load_jam_audio(jam_in=None, audio_file=audio_path)
            original_file_name = audio_path.split("/")[-1].split(".")[0]
            style = audio_path.split("/")[-2]
            audio_output_dir = os.path.join(
                self.config["augmented_audio_save_path"], style
            )
            jams_output_dir = os.path.join(
                self.config["augmented_meta_save_path"], style
            )

            if not os.path.exists(audio_output_dir):
                os.makedirs(audio_output_dir)
            if not os.path.exists(jams_output_dir):
                os.makedirs(jams_output_dir)

            for i, jam_out in enumerate(pipeline.transform(j_orig)):
                audio_output_path = os.path.join(
                    audio_output_dir, f"{original_file_name}_{i:02d}{output_format}"
                )
                jams_output_path = os.path.join(
                    jams_output_dir, f"{original_file_name}_{i:02d}.jams"
                )
                muda.save(audio_output_path, jams_output_path, jam_out)
