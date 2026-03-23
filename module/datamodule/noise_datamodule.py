import os
import re
import fnmatch
from pathlib import Path

import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .temporal_transforms import TemporalTransforms


class NoiseDataModule(LightningDataModule):
    """
    Dataset to load noise data.

    Supports wildcards and special keywords for train/val/test splits,
    and multiple audio extensions separated by '|'.
    Allows matching on full relative file paths (without extension).
    """

    def __init__(
        self,
        path_to_dataset: str,
        sr_standard: int = 16000,
        train_name=None,
        val_name=None,
        test_name=None,
        num_workers: int = 1,
        ext_audio="wav",
        pin_memory: bool = False,
        persistent_workers: bool = False,
        keep_data_in_memory: bool = False,
    ):
        super().__init__()
        self.path_to_dataset = Path(path_to_dataset)
        self.sr_standard = sr_standard
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.keep_data_in_memory = keep_data_in_memory

        # Parse ext_audio into a list, strip whitespace
        if isinstance(ext_audio, str):
            self.ext_audio_list = [e.strip() for e in re.split(r"\s*\|\s*", ext_audio.strip()) if e.strip()]
        elif isinstance(ext_audio, (list, tuple)):
            self.ext_audio_list = [str(e).strip() for e in ext_audio]
        else:
            raise ValueError("ext_audio must be a string or list of strings")

        # Recursively gather files matching any extension
        all_files = []
        for ext in self.ext_audio_list:
            all_files.extend(self.path_to_dataset.rglob(f"*.{ext}"))

        # Build map from relative path (no ext) to full Path
        self.key_to_path = {}
        for f in all_files:
            rel = f.relative_to(self.path_to_dataset).with_suffix("").as_posix()
            if rel not in self.key_to_path:
                self.key_to_path[rel] = f

        # Available keys for splitting
        all_keys = list(self.key_to_path.keys())
        self.available_keys = set(all_keys)

        # Process splits
        self.train_name = self._process_split_param(train_name, preselected=set())
        self.available_keys -= set(self.train_name)
        self.val_name = self._process_split_param(val_name, preselected=set(self.train_name))
        self.available_keys -= set(self.val_name)
        self.test_name = self._process_split_param(test_name, preselected=set(self.train_name) | set(self.val_name))
        self.available_keys -= set(self.test_name)

    def _process_split_param(self, param, preselected: set):
        """
        Handles split parameter logic.
        - list/tuple: match patterns first against full keys
        - None/'none': empty list
        - 'all'/'other': all keys excluding preselected
        - single pattern string: glob match against keys
        """
        all_keys = set(self.key_to_path.keys())

        # List/tuple: match patterns directly
        if isinstance(param, (list, tuple)):
            patterns = [str(p).strip() for p in param]
            matched = set()
            for pat in patterns:
                for key in all_keys:
                    if fnmatch.fnmatch(key, pat):
                        matched.add(key)
            return list(matched)

        # None or 'none'
        if param is None or (isinstance(param, str) and param.strip().lower() in ["none", ""]):
            return []

        # 'all' or 'other'
        if isinstance(param, str) and param.strip().lower() in ["all", "other"]:
            return list(all_keys - preselected)

        # Single string pattern
        if isinstance(param, str):
            pat = param.strip()
            matched = set()
            for key in all_keys:
                if fnmatch.fnmatch(key, pat):
                    matched.add(key)
            return list(matched - preselected)

        raise ValueError("Split parameter must be None, str, or list of str")

    def setup(self, stage=None) -> None:
        """
        Instantiate datasets for each split.
        """
        self.train_set = NoiseDataset(
            keys=self.train_name,
            key_to_path=self.key_to_path,
            sr_standard=self.sr_standard,
            keep_data_in_memory=self.keep_data_in_memory,
        )
        self.val_set = NoiseDataset(
            keys=self.val_name,
            key_to_path=self.key_to_path,
            sr_standard=self.sr_standard,
            keep_data_in_memory=self.keep_data_in_memory,
        )
        self.test_set = NoiseDataset(
            keys=self.test_name,
            key_to_path=self.key_to_path,
            sr_standard=self.sr_standard,
            keep_data_in_memory=self.keep_data_in_memory,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @property
    def info(self):
        return {
            "train_name": self.train_name,
            "val_name": self.val_name,
            "test_name": self.test_name,
            "num_workers": self.num_workers,
            "sr_standard": self.sr_standard,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }


class NoiseDataset(Dataset):
    """
    Dataset for noise data, loading audio from provided keys.
    """

    def __init__(
        self,
        keys,
        key_to_path: dict,
        sr_standard: int = 16000,
        keep_data_in_memory: bool = True,
        return_file_name: bool = False,
    ) -> None:
        self.keys = keys
        self.key_to_path = key_to_path
        self.sr_standard = sr_standard
        self.keep_data_in_memory = keep_data_in_memory
        self.return_file_name = return_file_name

        if self.keep_data_in_memory:
            self.cache = [None] * len(self)

    def __len__(self):
        return len(self.keys)

    def load_item(self, idx):
        key = self.keys[idx]
        file_path = self.key_to_path.get(key)
        if file_path is None:
            raise FileNotFoundError(f"No audio file found for key '{key}'")
        audio, sr = torchaudio.load(str(file_path), normalize=True)
        tt = TemporalTransforms(audio, sr, dims={"noise": [0]})
        tt.resampling(self.sr_standard)
        tt.normalize(dim="noise")
        if self.return_file_name:
            return tt.audio("noise"), key
        return tt.audio("noise")

    def __getitem__(self, idx):
        if self.keep_data_in_memory:
            if self.cache[idx] is None:
                self.cache[idx] = self.load_item(idx)
            return self.cache[idx]
        return self.load_item(idx)