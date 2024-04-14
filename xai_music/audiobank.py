import os
import pandas as pd
from pandas import DataFrame
import requests
from tqdm import tqdm
import numpy as np
from numpy import ndarray
import soundfile as sf
import librosa
from pathlib import Path
import os
import yaml
import json
from matplotlib import ticker
import matplotlib.pyplot as plt


class XaiMusic():

    def __init__(self, config_file: str, audio_format: str = '.wav'):
        self.config = self.__read_config_file(config_file)
        self.audio_format = audio_format

        # Check if the config file is valid
        self.__check_config_validity()

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
                f"The configuration file {config_file} does not exist.")
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)

    def __check_config_validity(self):
        """
        Checks if the necessary keys exist in the configuration file.

        Raises:
            KeyError: If any of the required keys are missing from the configuration.
        """
        required_keys = ['augmented_audio_save_path',
                         'mir_dataset_path', 'tempo_factor', 'keys', 'model_tag_result_save_path']
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise KeyError(
                f"Missing required configuration keys: {', '.join(missing_keys)}")

    def __time_stretch(self,
                       signal: ndarray,
                       time_stretch_rate: float) -> ndarray:
        """
        Time Stretch
        This function implements tempo modification.
        Args:
            signal (ndarray): The input audio signal to be processed.
            time_stretch_rate (float): The rate factor by which the audio signal's time is stretched. A value greater than 1 speeds up the playback, and a value less than 1 slows it down.

        Returns:
            ndarray: The time-stretched audio signal.
        """
        return librosa.effects.time_stretch(y=signal, rate=time_stretch_rate)

    def __pitch_scale(self,
                      signal: ndarray,
                      sr: float,
                      num_semitones: float) -> ndarray:
        """Pitch Shift
        This function implements pitch modification.

        Args:
            signal (ndarray): The input audio signal to be processed.
            sr (float): The sample rate of the audio signal.
            num_semitones (float): The number of semitones to shift the pitch by. A positive value raises the pitch, and a negative value lowers it.

        Returns:
            ndarray: The pitch-shifted audio signal.
        """
        return librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=num_semitones)

    def download_file_from_url(self, url: str, save_path: str):
        """ Download a file from a specified URL and save it to a local file path with a progress bar.

        Args:
            url (str): The URL of the file to download.
            save_path (str): The local path where the file should be saved.

        Returns:
            None
        """
        try:
            response = requests.get(url, stream=True)
            total_size_in_bytes = int(
                response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes,
                                unit='iB', unit_scale=True)
            with open(save_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                raise Exception("ERROR, something went wrong")
        except Exception as e:
            raise Exception(f"Failed to download file from {url}. Error: {e}")

    def __get_audio_file_paths(self, dataset_path: str) -> list:
        """
        Get a list of paths to audio files within the specified dataset path.
        Args:
            dataset_path (str): The path to the dataset containing audio files.
        Returns:
            list: A list of paths to the audio files.
        """
        audio_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith((".mp3", ".wav", ".flac")):
                    audio_files.append(os.path.join(root, file))
        return audio_files

    def synthesis_audiobank(self,
                            n_sample_genre: int = None) -> None:
        """
        Synthesize an augmented dataset from the music dataset.
        Args:
            n_sample_genre (int, optional): The number of samples from each genre to use. If None, all samples are used.
        Returns:
            None
        Raises:
            ValueError: If n_sample_genre is not a positive integer less than or equal to the number of unique genres.
        """
        audio_file_paths = self.__get_audio_file_paths(
            self.config['mir_dataset_path'])
        split_audio_file_paths = [m.split('/') for m in audio_file_paths]
        df_music_types = pd.DataFrame(
            [p[-2:] for p in split_audio_file_paths], columns=['type', 'audio'])

        # Prepare the list of audio file paths to be used for augmentation
        audio_lst = list()
        if n_sample_genre is None:
            audio_lst = [os.path.join(self.config['mir_dataset_path'],
                                      row.type,
                                      row.audio)
                         for row in df_music_types.itertuples(index=False)]
        else:
            if n_sample_genre <= 0 or n_sample_genre > df_music_types['type'].nunique():
                raise ValueError(
                    "n_sample_genre must be a positive integer less than or equal to the number of unique genres.")
            else:
                music_types = df_music_types['type'].unique().tolist()
                for mt in music_types:
                    tm_lst = df_music_types.loc[df_music_types['type'] == mt].sample(
                        n=n_sample_genre)
                    audio_lst.extend([os.path.join(self.config['mir_dataset_path'],
                                                   row.type,
                                                   row.audio)
                                     for row in tm_lst.itertuples(index=False)])
        # Create the augmented dataset
        for i, wa in enumerate(tqdm(audio_lst, desc="Processing audio files")):
            # Load the audio as a waveform `y`
            # Store the sampling rate as `sr`
            y, sr = librosa.load(wa)

            # With different tempo
            # scale (in this case stretch) the overall tempo by this factor
            for tmp_n, tmp_v in self.config['tempo_factor'].items():
                augments_1_audio = self.__time_stretch(y, tmp_v)

                # With different key
                save_path = Path(self.config['augmented_audio_save_path'])
                save_path.mkdir(parents=True, exist_ok=True)
                m_type = wa.split('/')[-2]
                base_file_name = wa.split('/')[-1][:-4]

                for key in self.config['keys']:
                    augments_2_audio = self.__pitch_scale(
                        augments_1_audio,
                        sr,
                        key)
                    key_note = 'original' if key == 0 else f"{key}"
                    file_name = f"{m_type}-{base_file_name}|{tmp_n}|{key_note}{self.audio_format}"
                    sf.write(os.path.join(save_path, file_name),
                             augments_2_audio,
                             sr)

    def save_tag_result(self,
                        pred_audio_path: str,
                        pred_tags: list) -> None:
        """
        Save the predicted tags for an audio file to a JSON file.
        Args:
            pred_audio_path (str): The path to the audio file for which the tags are predicted.
            pred_tags (list): The predicted tags for the audio file.
        Returns:
            None
        Raises:
            FileNotFoundError: If the audio file does not exist.
            TypeError: If the predicted tags are not a list.
        """
        if not os.path.exists(pred_audio_path):
            raise FileNotFoundError(
                f"The audio file {pred_audio_path} does not exist.")

        if not isinstance(pred_tags, list):
            raise TypeError("pred_tags must be a list from your model.")

        audio_name = os.path.basename(pred_audio_path)
        music, tempo, key = os.path.splitext(audio_name)[0].split('|')
        audio_dict = {'tags': pred_tags,
                      'music': music,
                      'tempo': tempo,
                      'key': key}
        result_file_path = os.path.join(
            self.config['model_tag_result_save_path'], 'tag_result.json')
        tag_result_dir = os.path.dirname(result_file_path)

        # Ensure the file and its directory exists
        os.makedirs(tag_result_dir, exist_ok=True)
        if not os.path.exists(result_file_path):
            with open(result_file_path, 'w') as file:
                file.write('{}')  # Create an empty JSON file

        # Open the file in 'r+' mode to read and write. If the file does not exist, it's created.
        with open(result_file_path, 'r+') as file:
            try:
                audio_json = json.load(file)
            except json.JSONDecodeError:
                audio_json = {}
            audio_json[os.path.splitext(audio_name)[0]] = audio_dict
            file.seek(0)
            file.truncate()
            json.dump(audio_json, file, indent=4)

    def tag_analysis(self):
        result_file_path = os.path.join(
            self.config['model_tag_result_save_path'], 'tag_result.json')

        if not os.path.exists(result_file_path):
            raise FileNotFoundError(f"File {result_file_path} does not exist.")
        with open(result_file_path, 'r') as file:
            df = pd.read_json(file)

        # Transform tables
        df = df.T

        # Get list of music to analyze
        music_list = df['music'].unique().tolist()

        # Tag analysis by tempo change
        for music in tqdm(music_list, desc="Analyzing Music Tags by tempo change"):
            music_df = df[df['music'] == music]
            ohe_df = music_df.drop(columns=['tags']).join(
                music_df.tags.str.join('|').str.get_dummies())

            self.__plot_result(plt_save_path=music,
                               plt_save_name='tag_analysis_by_tempo.png',
                               xlabel='Predicted tag in different tempos',
                               group_ohe_df=ohe_df.groupby(['tempo', 'key']).sum().unstack('tempo'))

            ohe_df.drop(columns=['music', 'key'], inplace=True)
            ohe_df = ohe_df.groupby(['tempo']).sum()
            self.__plot_result(plt_save_path=music,
                               plt_save_name='tag_analysis_by_tempo_sum.png',
                               xlabel='Sum(Total Amount of Predicted tags)',
                               group_ohe_df=ohe_df)

        # Tag analysis by key change
        for music in tqdm(music_list, desc="Analyzing Music Tags by key change"):
            music_df = df[df['music'] == music]
            ohe_df = music_df.drop(columns=['tags']).join(
                music_df.tags.str.join('|').str.get_dummies())

            self.__plot_result(plt_save_path=music,
                               plt_save_name='tag_analysis_by_key.png',
                               xlabel='Predicted tag in different keys',
                               group_ohe_df=ohe_df.groupby(['key', 'tempo']).sum().unstack('key'))

            ohe_df.drop(columns=['music', 'tempo'], inplace=True)
            ohe_df = ohe_df.groupby(['key']).sum()
            self.__plot_result(plt_save_path=music,
                               plt_save_name='tag_analysis_by_key_sum.png',
                               xlabel='Sum(Total Amount of Predicted tags)',
                               group_ohe_df=ohe_df)

    def __plot_result(self,
                      group_ohe_df: DataFrame,
                      plt_save_path: str,
                      plt_save_name: str,
                      xlabel: str = 'tag',
                      ) -> None:
        """Generates and saves a plot based on the one-hot encoded DataFrame.

        This function takes a DataFrame that has been one-hot encoded based on tags associated with music tracks. It generates a heatmap plot from this data, showing the distribution of tags across different tempos. The plot is then saved to the specified path with the given file name.

        Args:
            plt_save_path (str): The directory path where the plot will be saved.
            plt_save_name (str): The name of the file in which the plot will be saved.
            group_ohe_df (DataFrame): The one-hot encoded DataFrame containing tag distribution data.
            xlabel (str): The label for the x-axis.
        """
        # Ensure all data in group_ohe_df is numeric
        group_ohe_df = group_ohe_df \
            .apply(pd.to_numeric, errors='coerce', downcast='integer') \
            .fillna(0).astype(int)

        fig, ax = plt.subplots()
        fig.set_figwidth(20)

        arr = group_ohe_df.to_numpy()
        key, tag = arr.shape

        x_labels_list = [col for col in group_ohe_df.columns]
        y_labels_list = group_ohe_df.index.tolist()

        ax.matshow(arr, cmap=plt.cm.Blues)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(group_ohe_df.index.name)

        # Adjust the FixedLocator to match the number of labels including the empty one
        ax.xaxis.set_major_locator(
            ticker.FixedLocator(np.arange(-0.5, len(x_labels_list), 1)))  # Adjusted for x-axis
        ax.yaxis.set_major_locator(
            ticker.FixedLocator(np.arange(-0.5, len(y_labels_list), 1)))  # Adjusted for y-axis
        ax.set_xticklabels([''] + x_labels_list, rotation=90)
        ax.set_yticklabels([''] + y_labels_list)

        # Loop to add text inside the plot
        for i in range(tag):
            for j in range(key):
                c = arr[j, i]
                ax.text(i, j, str(c), va='center', ha='center')

        save_dir = os.path.join(
            self.config['model_tag_result_save_path'], plt_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, plt_save_name))
