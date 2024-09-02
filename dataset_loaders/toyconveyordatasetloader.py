# This class is responsible for loading the ToyADMOS dataset, both from disk initially and later at different sample rates and preprocessing features. The class also contains the logic for splitting the dataset into training, validation and test sets.

# Standard Library Imports
import copy
import math
from typing import Optional, Any

# Third Party Imports
import librosa
import tensorflow as tf
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

# Local Imports
import datasetloader
import data


class ToyConveyorDatasetLoader(datasetloader.DatasetLoader):
    # We do the initial part of the dataset loading in the constructor to avoid loading from files continiously
    # There are 1800 normal files and 400 anomalous files (in channel 1)
    def __init__(
        self,
        file_path: str,
        num_files: list[int],
        dataset_options: dict[str, Any],
        num_cores_to_use: int,
    ) -> None:
        self.num_cores_to_use = num_cores_to_use

        # If no case is specified, use case1
        if dataset_options.get("case") == None:
            dataset_options["case"] = "case1"

        # If no channel is specified, use channel 1
        if dataset_options.get("channel") == None:
            dataset_options["channel"] = "1"

        # If no audio-seconds-to-load is specified, use 10 seconds
        if dataset_options.get("audio_seconds_to_load") == None:
            dataset_options["audio_seconds_to_load"] = 10

        # If no sound gain is specified, use 1
        if dataset_options.get("sound_gain") == None:
            dataset_options["sound_gain"] = 1

        # If no noise gain is specified, use 1
        if dataset_options.get("noise_gain") == None:
            dataset_options["noise_gain"] = 1

        # Find all files in the requested channel directory
        normal_files = tf.io.gfile.glob(
            f"{file_path}/{dataset_options.get('case')}/NormalSound_IND/*ch{dataset_options.get('channel')}*.wav"
        )
        anomalous_files = tf.io.gfile.glob(
            f"{file_path}/{dataset_options.get('case')}/AnomalousSound_IND/*ch{dataset_options.get('channel')}*.wav"
        )
        noise_files = tf.io.gfile.glob(
            f"{file_path}/EnvironmentalNoise_CNT/*{dataset_options.get('case')}*ch{dataset_options.get('channel')}*.wav"
        )

        if len(normal_files) == 0:
            raise FileNotFoundError(
                "Normal sound files not found, please change dataset options to point to a directory containing the ToyADMOS normal sound"
            )
        if len(anomalous_files) == 0:
            raise FileNotFoundError(
                "Anomalous sound files not found, please change dataset options to point to a directory containing the ToyADMOS anomalous sound"
            )
        if len(noise_files) == 0:
            raise FileNotFoundError(
                "Noise sound files not found, please change dataset options to point to a directory containing the ToyADMOS noise sound"
            )

        # Cut the amount of processed files according to the program parameters
        normal_files = normal_files[: num_files[0]]
        anomalous_files = anomalous_files[: num_files[1]]
        # We should have as many noise files as the sum of normal and anomalous files
        noise_files = noise_files[: num_files[0] + num_files[1]]

        # Get the sample rate of the audio files. The sample rate is assumed to be the same for every file.
        self.base_sr = librosa.get_samplerate(normal_files[0])

        # Load the audio files using librosa. Use multiple python processes in order to speed up loading.
        base_normal_audio = Parallel(n_jobs=self.num_cores_to_use)(
            delayed(self._librosa_load_without_sample_rate)(
                file, self.base_sr, dataset_options.get("audio_seconds_to_load")
            )
            for file in normal_files
        )
        base_anomalous_audio = Parallel(n_jobs=self.num_cores_to_use)(
            delayed(self._librosa_load_without_sample_rate)(
                file, self.base_sr, dataset_options.get("audio_seconds_to_load")
            )
            for file in anomalous_files
        )
        base_noise_audio = Parallel(n_jobs=self.num_cores_to_use)(
            delayed(self._librosa_load_without_sample_rate)(
                file, self.base_sr, dataset_options.get("audio_seconds_to_load")
            )
            for file in noise_files
        )

        if (
            (base_normal_audio is None)
            or (base_anomalous_audio is None)
            or (base_noise_audio is None)
        ):
            raise TypeError("One of the audio files is failed to load.")

        # Make sure that base_noise_audio is at least the length of the sum of the two others:
        num_duplicates = (len(base_normal_audio) + len(base_anomalous_audio)) / len(
            base_noise_audio
        )

        base_noise_audio = base_noise_audio * math.ceil(num_duplicates)

        # Mix noise into the other audio files.
        normal_noise_audio = base_noise_audio[: num_files[0]]
        anomalous_noise_audio = base_noise_audio[num_files[1]]

        normal_noise_zip = zip(base_normal_audio, normal_noise_audio)
        anomalous_noise_zip = zip(base_anomalous_audio, anomalous_noise_audio)

        self.base_normal_audio = Parallel(n_jobs=self.num_cores_to_use)(
            delayed(self._mix_audio)(
                audio,
                dataset_options.get("sound_gain"),
                noise,
                dataset_options.get("noise_gain"),
            )
            for audio, noise in normal_noise_zip  # type: ignore - to my understanding a for loop should take an iterator which zip is, not require an interable.
        )
        self.base_anomalous_audio = Parallel(n_jobs=self.num_cores_to_use)(
            delayed(self._mix_audio)(
                audio,
                dataset_options.get("sound_gain"),
                noise,
                dataset_options.get("noise_gain"),
            )
            for audio, noise in anomalous_noise_zip  # type: ignore - same reason as above
        )

    def supervised_dataset(
        self, input_data: tuple[list, ...], test_size: float = 0.2
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Assign numeric labels to normal and abnormal samples
        i = 0
        y = []
        for d in input_data:
            y.append(tf.fill(len(d), i))
            i += 1

        # Combine the two feature and two label lists into one feature and one label tensor
        X = tf.concat(input_data, 0)
        y = tf.concat(y, 0)

        X_train, X_test, y_train, y_test = train_test_split(
            X.numpy(), y.numpy(), test_size=test_size, stratify=y  # type: ignore - it seems that typing in tensorflow do not return very useful types.
        )

        return data.Data(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    # This method is called to load the dataset according to a specific data configuration.
    # The base dataset (full granularity) should already have been loaded in the constructor of this class.
    def configure_dataset(
        self,
        **kwargs: dict[str, Any],
    ) -> tuple[list, list]:
        if (self.base_normal_audio is None) or (self.base_anomalous_audio is None):
            raise TypeError("The base audio files were not loaded correctly.")

        # Check that the frame_size, hop_length, num_mel_filters and num_mfccs are specified.
        if kwargs.get("frame_size") is None:
            raise ValueError(
                "The frame_size option must be specified for the toyconveyor dataset."
            )
        if kwargs.get("hop_length") is None:
            raise ValueError(
                "The hop_length option must be specified for the toyconveyor dataset."
            )

        normal_audio = copy.deepcopy(self.base_normal_audio)
        anomalous_audio = copy.deepcopy(self.base_anomalous_audio)

        normal_audio = Parallel(n_jobs=self.num_cores_to_use)(
            delayed(librosa.resample)(
                audio,
                orig_sr=self.base_sr,
                target_sr=kwargs["sample_rate"],
                res_type="kaiser_fast",
            )
            for audio in normal_audio
        )
        anomalous_audio = Parallel(n_jobs=self.num_cores_to_use)(
            delayed(librosa.resample)(
                audio,
                orig_sr=self.base_sr,
                target_sr=kwargs["sample_rate"],
                res_type="kaiser_fast",
            )
            for audio in anomalous_audio
        )

        if (normal_audio is None) or (anomalous_audio is None):
            raise TypeError("Resampling audio files failed.")

        match kwargs["audio_representation"]:
            case "waveform":
                # While we can easily load data as a waveform it can not be processed by a standard convolutional block that expects image like dimensions.
                raise NotImplementedError(
                    "Loading data as a waveform has not been implemented yet"
                )
                return (normal_audio, anomalous_audio)
            case "spectrogram":
                normal_spectogram = Parallel(n_jobs=self.num_cores_to_use)(
                    delayed(self._create_spectrogram)(
                        audio, kwargs.get("frame_size"), kwargs.get("hop_length")
                    )
                    for audio in normal_audio
                )
                anomalous_spectrograms = Parallel(n_jobs=self.num_cores_to_use)(
                    delayed(self._create_spectrogram)(
                        audio, kwargs.get("frame_size"), kwargs.get("hop_length")
                    )
                    for audio in anomalous_audio
                )

                if (normal_spectogram is None) or (anomalous_spectrograms is None):
                    raise TypeError("Creating spectrograms failed.")

                return (normal_spectogram, anomalous_spectrograms)
            case "mel_spectrogram":
                normal_mel_spectogram = Parallel(n_jobs=self.num_cores_to_use)(
                    delayed(self._create_mel_spectrogram)(
                        audio,
                        kwargs.get("sample_rate"),
                        kwargs.get("frame_size"),
                        kwargs.get("hop_length"),
                        kwargs.get("num_mel_filters"),
                    )
                    for audio in normal_audio
                )
                anomalous_mel_spectrograms = Parallel(n_jobs=self.num_cores_to_use)(
                    delayed(self._create_mel_spectrogram)(
                        audio,
                        kwargs.get("sample_rate"),
                        kwargs.get("frame_size"),
                        kwargs.get("hop_length"),
                        kwargs.get("num_mel_filters"),
                    )
                    for audio in anomalous_audio
                )

                if (normal_mel_spectogram is None) or (
                    anomalous_mel_spectrograms is None
                ):
                    raise TypeError("Creating mel-spectrograms failed.")

                return (normal_mel_spectogram, anomalous_mel_spectrograms)
            case "mfcc":
                normal_mfccs = Parallel(n_jobs=self.num_cores_to_use)(
                    delayed(self._create_mfcc)(
                        audio,
                        kwargs.get("sample_rate"),
                        kwargs.get("frame_size"),
                        kwargs.get("hop_length"),
                        kwargs.get("num_mel_filters"),
                        kwargs.get("num_mfccs"),
                    )
                    for audio in normal_audio
                )
                anomalous_mfccs = Parallel(n_jobs=self.num_cores_to_use)(
                    delayed(self._create_mfcc)(
                        audio,
                        kwargs.get("sample_rate"),
                        kwargs.get("frame_size"),
                        kwargs.get("hop_length"),
                        kwargs.get("num_mel_filters"),
                        kwargs.get("num_mfccs"),
                    )
                    for audio in anomalous_audio
                )

                if (normal_mfccs is None) or (anomalous_mfccs is None):
                    raise TypeError("Creating mfccs failed.")

                return (normal_mfccs, anomalous_mfccs)
            case _:
                raise NotImplementedError(
                    "This dataloader only supports loading audio as spectrograms, mel_spectrograms and mfccs."
                )

    def _librosa_load_without_sample_rate(
        self, file: str, sr: float, duration: float
    ) -> np.ndarray:
        audio, _ = librosa.load(file, sr=sr, duration=duration)
        return audio

    def _mix_audio(
        self, audio: np.ndarray, audio_gain: float, noise: np.ndarray, noise_gain: float
    ) -> np.ndarray:
        return (audio * audio_gain) + (noise * noise_gain)

    def _create_spectrogram(
        self, audio_sample: np.ndarray, frame_size: int, hop_length: int
    ) -> np.ndarray:
        # |stft|^2 is the power spectrogram, |stft| is the amplitude spectrogram
        # We can convert both into the log-amplitude spectrogram using librosa.power_to_db or librosa.amplitude_to_db respectivly
        # It is unclear whether we should do a log representation of both amplitude and frequence in a standard spectrogram.
        # In this function we only do a log representation of amplitude.
        # We always return the spectrograms with an additional axis as this is expected by tensorflows convolutional layers (since images have a channel dimension)
        return librosa.power_to_db(
            np.abs(librosa.stft(audio_sample, n_fft=frame_size, hop_length=hop_length))
            ** 2
        )[..., tf.newaxis]

    def _create_mel_spectrogram(
        self,
        audio_sample: np.ndarray,
        sample_rate: float,
        frame_size: int,
        hop_length: int,
        num_mel_filters: int,
    ) -> np.ndarray:
        # Check that the num_mel_filters is not None
        if num_mel_filters is None:
            raise ValueError(
                "The number of mel filters must be specified when creating a mel spectrogram."
            )
        # Unlike for the spectrogram, the mel spectrogram has a directly accessible function in librosa.
        return librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=audio_sample,
                sr=sample_rate,
                n_fft=frame_size,
                hop_length=hop_length,
                n_mels=num_mel_filters,
            )
        )[..., tf.newaxis]

    def _create_mfcc(
        self,
        audio_sample: np.ndarray,
        sample_rate: float,
        frame_size: int,
        hop_length: int,
        num_mel_filters: int,
        num_mfccs: int,
    ) -> np.ndarray:
        # Check that the num_mel_filters and num_mfccs is not None
        if num_mel_filters is None:
            raise ValueError(
                "The number of mel filters must be specified when creating a mfcc."
            )
        if num_mfccs is None:
            raise ValueError(
                "The number of mfccs must be specified when creating a mfcc."
            )
        # Maybe also do first and second derivatives - is supposedly improving accuracy
        # Is often not used that much in deep learning, and is made to understand speech and music - not machine sounds.
        mfcc = librosa.feature.mfcc(
            y=audio_sample,
            sr=sample_rate,
            n_fft=frame_size,
            hop_length=hop_length,
            n_mels=num_mel_filters,
            n_mfcc=num_mfccs,
        )
        delta_mfcc = librosa.feature.delta(mfcc, mode="nearest")
        delta2_mfcc = librosa.feature.delta(mfcc, order=2, mode="nearest")
        return np.concatenate((mfcc, delta_mfcc, delta2_mfcc))[..., tf.newaxis]
