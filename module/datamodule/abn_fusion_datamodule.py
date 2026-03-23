import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from .temporal_transforms import TemporalTransforms
from .ab_datamodule import AB_DataModule


class FusionDataModule(LightningDataModule):
    """
    Dataset to fuse clean audio with noise.
    Return format: (ac, bc, noise_ac)
    """

    def __init__(
        self,
        ab_datamodule,
        noise_datamodule,
        snr=None,
        train_snr=None,
        val_snr=None,
        test_snr=None,
        bs_train=16,
        bs_val=8,
        bs_test=1,
        num_workers=4,
        ratio=None,
        keep_clean_data=False,
        pin_memory=True,
        persistent_workers=True,
        is_smoothing=True,
    ):
        """
        Args:
        -----------
            ab_datamodule: LightningDataModule
                Data module that loads the clean audio
            noise_datamodule: LightningDataModule
                Data module that loads the noise
            snr: int or list of int, default None
                Signal to noise ratio
            train_snr: int or tuple of int or list of int, default None
                Signal to noise ratio for training dataset
            val_snr: int or tuple of int or list of int, default None
                Signal to noise ratio for validation dataset
            test_snr: int or tuple of int or list of int, default None
                Signal to noise ratio for test dataset
            bs_train: int, default 16
                Batch size for training dataset
            bs_val: int, default 8
                Batch size for validation dataset
            num_workers: int, default 4
                Number of workers for data loader
            ratio: int, None
                Number of samples to keep from the original dataset by one sample (Only take effect when snr is tuple)
            keep_clean_data: bool, default True
                Keep the clean audio in the dataset
        """
        super().__init__()
        if isinstance(ab_datamodule, dict):
            from ..utils.utils import initialize_datamodule_from_dict

            self.ab_datamodule = initialize_datamodule_from_dict(ab_datamodule)
        else:
            self.ab_datamodule = ab_datamodule
        if isinstance(noise_datamodule, dict):
            from ..utils.utils import initialize_datamodule_from_dict

            self.noise_datamodule = initialize_datamodule_from_dict(noise_datamodule)
        else:
            self.noise_datamodule = noise_datamodule
        if not hasattr(self.ab_datamodule, "sr_standard"):
            raise ValueError("ab_datamodule must have sr_standard attribute")
        if not hasattr(self.noise_datamodule, "sr_standard"):
            raise ValueError("noise_datamodule must have sr_standard attribute")
        if self.ab_datamodule.sr_standard != self.noise_datamodule.sr_standard:
            raise ValueError("Sampling rate must be the same")
        self.train_snr = None
        self.val_snr = None
        self.test_snr = None
        if snr is not None:
            if isinstance(snr, int):
                snr = [snr]
            if isinstance(snr, list) and len(snr) == 0:
                raise ValueError("snr list must have at least 1 element")
            if isinstance(snr, list) and snr[0] == "interval":
                snr = tuple(snr[1:])
            if isinstance(snr, tuple) and len(snr) != 2:
                raise ValueError("snr tuple must have 2 elements, means an interval")
            self.train_snr = snr
            self.val_snr = snr
            self.test_snr = snr
        if train_snr is not None:
            if isinstance(train_snr, int):
                train_snr = [train_snr]
            if isinstance(train_snr, list) and len(train_snr) == 0:
                raise ValueError("snr list must have at least 1 element")
            if isinstance(train_snr, list) and train_snr[0] == "interval":
                train_snr = tuple(train_snr[1:])
            if isinstance(train_snr, tuple) and len(train_snr) != 2:
                raise ValueError(
                    "train_snr tuple must have 2 elements, means an interval"
                )
            self.train_snr = train_snr
        if val_snr is not None:
            if isinstance(val_snr, int):
                val_snr = [val_snr]
            if isinstance(val_snr, list) and len(val_snr) == 0:
                raise ValueError("snr list must have at least 1 element")
            if isinstance(val_snr, list) and val_snr[0] == "interval":
                val_snr = tuple(val_snr[1:])
            if isinstance(val_snr, tuple) and len(val_snr) != 2:
                raise ValueError(
                    "val_snr tuple must have 2 elements, means an interval"
                )
            self.val_snr = val_snr
        if test_snr is not None:
            if isinstance(test_snr, int):
                test_snr = [test_snr]
            if isinstance(test_snr, list) and len(test_snr) == 0:
                raise ValueError("snr list must have at least 1 element")
            if isinstance(test_snr, list) and test_snr[0] == "interval":
                test_snr = tuple(test_snr[1:])
            if isinstance(test_snr, tuple) and len(test_snr) != 2:
                raise ValueError(
                    "test_snr tuple must have 2 elements, means an interval"
                )
            self.test_snr = test_snr
        self.bs_train = bs_train
        self.bs_val = bs_val
        self.bs_test = bs_test
        self.num_workers = num_workers
        self.ratio = ratio
        self.keep_clean_data = keep_clean_data
        if self.ratio is None:
            if isinstance(self.train_snr, tuple):
                raise ValueError("ratio must be defined when train_snr is tuple")
            if isinstance(self.val_snr, tuple):
                raise ValueError("ratio must be defined when val_snr is tuple")
            if isinstance(self.test_snr, tuple):
                raise ValueError("ratio must be defined when test_snr is tuple")
        elif self.ratio <= 0:
            raise ValueError("ratio must be greater than 0")
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.is_smoothing = is_smoothing

    def setup(self, stage=None):
        self.ab_datamodule.setup(stage)
        self.noise_datamodule.setup(stage)
        self.train_set = FusionDataset(
            self.ab_datamodule.train_set,
            self.noise_datamodule.train_set,
            self.train_snr,
            self.ratio,
            self.keep_clean_data,
            self.is_smoothing,
        )
        self.val_set = FusionDataset(
            self.ab_datamodule.val_set,
            self.noise_datamodule.val_set,
            self.val_snr,
            self.ratio,
            self.keep_clean_data,
            self.is_smoothing,
        )
        self.test_set = FusionDataset(
            self.ab_datamodule.test_set,
            self.noise_datamodule.test_set,
            self.test_snr,
            self.ratio,
            self.keep_clean_data,
            self.is_smoothing,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.bs_train,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.bs_val,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.bs_test,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @property
    def info(self):
        return {
            "ab_datamodule": self.ab_datamodule.info,
            "noise_datamodule": self.noise_datamodule.info,
            "train_snr": self.train_snr,
            "val_snr": self.val_snr,
            "test_snr": self.test_snr,
            "bs_train": self.bs_train,
            "bs_val": self.bs_val,
            "bs_test": self.bs_test,
            "num_workers": self.num_workers,
            "ratio": self.ratio,
            "keep_clean_data": self.keep_clean_data,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }


class FusionDataset(Dataset):
    def __init__(
        self,
        ab_dataset,
        noise_dataset,
        snr,
        ratio,
        keep_clean_data,
        is_smoothing: bool = True,
        return_file_name: bool = False,
    ):
        self.ab_dataset = ab_dataset
        self.noise_dataset = noise_dataset
        self.snr = snr
        self.ratio = ratio
        self.keep_clean_data = keep_clean_data
        self.len_data = len(self.ab_dataset)
        self.len_noise = len(self.noise_dataset)
        if isinstance(self.snr, list):
            self.ratio = None
        self.is_smoothing = is_smoothing
        self.return_file_name = return_file_name

    def __len__(self):
        if isinstance(self.snr, tuple):
            return int(self.len_data * self.ratio)
        elif isinstance(self.snr, list):
            return self.len_data * (
                self.len_noise * len(self.snr) + self.keep_clean_data
            )
        else:
            raise ValueError("snr must be tuple or list")

    def __getitem__(self, idx):
        if idx >= len(self) or idx < 0:
            raise IndexError("Index out of range")
        if self.ratio is None:
            ab_idx = idx // (self.len_noise * len(self.snr) + self.keep_clean_data)
            rnd_idx = idx % (self.len_noise * len(self.snr) + self.keep_clean_data)
        else:
            ab_idx = torch.randint(0, self.len_data, (1,)).item()
            rnd_idx = torch.randint(
                0, self.len_noise * len(self.snr) + self.keep_clean_data, (1,)
            ).item()
        if self.return_file_name:
            ac, bc, audio_name = self.ab_dataset[ab_idx]
        else:
            ac, bc = self.ab_dataset[ab_idx]
        sr = self.ab_dataset.sr_standard
        noise_ac = TemporalTransforms(
            audio=ac.clone(),
            sr=sr,
            dims={"noise_ac": [0]},
        )
        if rnd_idx == self.len_noise * len(self.snr):
            noise = None
            snr = None
        else:
            noise_idx = rnd_idx % self.len_noise
            if self.return_file_name:
                noise, noise_name = self.noise_dataset[noise_idx]
            else:
                noise = self.noise_dataset[noise_idx]   
            snr_idx = rnd_idx // self.len_noise
            noise_sr = self.noise_dataset.sr_standard
            if sr != noise_sr:
                noise = TemporalTransforms(
                    audio=noise, sr=noise_sr, dims={"noise": [0]}
                )
                noise.resampling(sr)
                noise = noise.audio("noise")
            if isinstance(self.snr, tuple):
                snr = self.snr[0] + (self.snr[1] - self.snr[0]) * torch.rand(1).item()
            else:
                snr = self.snr[snr_idx]
            noise_ac.add_noise(noise=noise, snr=snr, dim="noise_ac")
        if self.is_smoothing:
            noise_ac.smoothing(dim="noise_ac")
        noise_ac.normalize(dim="noise_ac")
        noise_ac = noise_ac.audio("noise_ac")
        if self.return_file_name:
            return ac, noise_ac, bc, audio_name, noise_name
        return ac, noise_ac, bc
