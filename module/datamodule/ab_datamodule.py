"""
Module to load AC_BC data and apply degradation.
If len_seconds is None, audio will not be cropped, and batch size will be set to 1.
dataloader return format: (ac, bc)
    ac: torch.Tensor
    bc: torch.Tensor
"""

import os
from pathlib import Path
from typing import Tuple

import torchaudio
from pytorch_lightning import LightningDataModule
from .temporal_transforms import TemporalTransforms
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import warnings

class AB_DataModule(LightningDataModule):
    """
    Custom ABCS LightningDataModule.
    Return format: (ac, bc)

    Args:
    -----------
        path_to_dataset: str, default None
            Folder that contains the dataset

        len_seconds_train: str, default None
            Sample length for train dataset in seconds

        len_seconds_val: str, default None
            Sample length for validation in seconds
        
        len_seconds_test: str, default None
            Sample length for test dataset in seconds

        bs_train: str, default None
            Batch size for train dataset

        bs_val: str, default None
            Batch size for validation datasets

        bs_test: str, default None
            Batch size for test dataset
        
        num_workers: int, default None
            Number of workers used to load data

        sr_standard: int, default True
            Sampling rate at which all samples are resample

        separator: str, default None
            ASCII character used in file names

        train_folder: str, default train/audio
            Folder that contains the train dataset

        val_folder: str, default dev/audio
            Folder that contains the validation dataset

        test_folder: str, default test/audio
            Folder that contains the test dataset

        pin_memory: bool, default True
            Pin memory for DataLoader
        
        persistent_workers: bool, default True
            Persistent workers for DataLoader
    """

    def __init__(
        self,
        path_to_dataset,
        sr_standard=16000,
        len_seconds_train=1,
        len_seconds_val=1,
        len_seconds_test=1,
        bs_train=16,
        bs_val=8,
        bs_test=1,
        num_workers=4,
        separator="_",
        train_folder="Audio/train",
        val_folder="Audio/dev",
        test_folder="Audio/test",
        is_smoothing=True,
        deterministic=False,
        pin_memory=True,
        persistent_workers=True,
    ):
        super().__init__()

        self.path_to_dataset = path_to_dataset
        self.bs_train = bs_train
        self.bs_val = bs_val
        self.num_workers = num_workers
        self.sr_standard = sr_standard
        self.len_seconds_train = len_seconds_train
        self.len_seconds_val = len_seconds_val
        self.len_seconds_test = len_seconds_test
        self.bs_test = bs_test
        if len_seconds_train is None and self.bs_train > 1:
            warnings.warn(
                "train length seconds is None and batch size > 1, batch size will be set to 1, and audio will not be cropped"
            )
            self.bs_train = 1
        if len_seconds_val is None and self.bs_val > 1:
            warnings.warn(
                "val length seconds is None and batch size > 1, batch size will be set to 1, and audio will not be cropped"
            )
            self.bs_val = 1
        if len_seconds_test is None and self.bs_test > 1:
            warnings.warn(
                "test length seconds is None and batch size > 1, batch size will be set to 1, and audio will not be cropped"
            )
            self.bs_test = 1
        self._separator = separator
        self._train_folder = train_folder
        self._val_folder = val_folder
        self._test_folder = test_folder
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.is_smoothing = is_smoothing
        self.deterministic = deterministic

    def setup(self, stage=None) -> None:
        """
        Things to do on every accelerator in distributed mode
        """
        self.train_set = AB_Dataset(
            path=self.path_to_dataset,
            folder=self._train_folder,
            deterministic=self.deterministic,
            sr_standard=self.sr_standard,
            len_seconds=self.len_seconds_train,
            separator=self._separator,
            is_smoothing=self.is_smoothing,
        )
        self.val_set = AB_Dataset(
            path=self.path_to_dataset,
            folder=self._val_folder,
            deterministic=self.deterministic,
            sr_standard=self.sr_standard,
            len_seconds=self.len_seconds_val,
            separator=self._separator,
            is_smoothing=self.is_smoothing,
        )
        self.test_set = AB_Dataset(
            path=self.path_to_dataset,
            folder=self._test_folder,
            deterministic=self.deterministic,
            sr_standard=self.sr_standard,
            len_seconds=self.len_seconds_test,
            separator=self._separator,
            is_smoothing=self.is_smoothing,
        )

    def train_dataloader(self):
        """
        This function creates the train DataLoader.
        """
        return DataLoader(
            self.train_set,
            batch_size=self.bs_train,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        """
        This function creates the validation DataLoader.
        """
        return DataLoader(
            self.val_set,
            batch_size=self.bs_val,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        """
        This function creates the test DataLoader.
        """
        return DataLoader(
            self.test_set,
            batch_size=self.bs_test,
            shuffle=False,
            num_workers=self.num_workers
        )

    @property
    def info(self):
        return {
            "path_to_dataset": self.path_to_dataset,
            "train_name": self._train_folder,
            "val_name": self._val_folder,
            "test_name": self._test_folder,
            "len_seconds_train": self.len_seconds_train,
            "len_seconds_val": self.len_seconds_val,
            "len_seconds_test": self.len_seconds_test,
            "bs_train": self.bs_train,
            "bs_val": self.bs_val,
            "bs_test": self.bs_test,
            "num_workers": self.num_workers,
            "sr_standard": self.sr_standard,
            "separator": self._separator,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }

class AB_Dataset(Dataset):
    """Create a Dataset for ABCS"""

    _ext_audio = ".wav"

    def __init__(
        self,
        path,
        folder,
        sr_standard,
        separator,
        len_seconds: float = 6.0,
        deterministic: bool = False,
        is_smoothing: bool = True,
        return_file_name: bool = False,
    ) -> None:
        self.len_seconds = len_seconds
        self.sr_standard = sr_standard
        self.determinist = deterministic
        self._separator = separator
        self._path = path
        self._folder = folder
        self._full_path = os.path.join(self._path, self._folder)
        self._walker = sorted(
            str(p.stem) for p in Path(self._full_path).glob("*/*" + self._ext_audio)
        )
        self.is_smoothing = is_smoothing
        self.return_file_name = return_file_name

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        """Load the n-th sample from the dataset and apply degradation

        Args:
            n:int,
                The index of the sample to be loaded

        Returns:
            audio:torch.Tensor,
                audio is shape [1,time_len]
            audio_corrupted:torch.Tensor,
                audio_corrupted is shape [1,time_len]
        """
        fileid = self._walker[n]
        audio, sr = self.load_item(fileid, self._full_path, self._ext_audio)

        tt_audio_ref = TemporalTransforms(
            audio, sr, deterministic=self.determinist,dims={"ac": [0],"bc":[1]}
        )
        tt_audio_ref.resampling(new_freq=self.sr_standard)
        if self.len_seconds is not None:
            tt_audio_ref.select_part(self.len_seconds)
        if self.is_smoothing:
            tt_audio_ref.smoothing(dim="ac")
            tt_audio_ref.smoothing(dim="bc")
        tt_audio_ref.normalize(dim="ac")
        tt_audio_ref.normalize(dim="bc")
        if self.return_file_name:
            return tt_audio_ref.audio('ac'), tt_audio_ref.audio('bc'), fileid
        return tt_audio_ref.audio('ac'), tt_audio_ref.audio('bc')

    def __len__(self) -> int:
        return len(self._walker)

    def load_item(
        self, fileid: str, path: str, ext_audio: str
    ) -> Tuple[Tensor, int]:
        speaker_id, chapter_id, utterance_id = fileid.split(self._separator, 2)
        fileid_audio = (
            speaker_id + self._separator + chapter_id + self._separator + utterance_id
        )
        file_audio = fileid_audio + ext_audio
        file_audio = os.path.join(path, speaker_id, file_audio)

        # Load audio
        waveform, sample_rate = torchaudio.load(
            file_audio, normalize=True
        )  # normalize
        return waveform, sample_rate
