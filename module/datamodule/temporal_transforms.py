""" Definition of several audio transformation methods in the TemporalTransforms class  """

import torch
from torchaudio.functional import lowpass_biquad
from torchaudio.transforms import Resample
from typing import Union


class TemporalTransforms(object):
    """
    Processing class to apply transforms on an audio signal.

    Args:
        audio: torch.Tensor,
            audio waveform to process

        sr: int,
            sampling rate of the audio

        padding_length: int,
            used for IRR response stabilisation

        deterministic: bool,
            whether to apply a deterministic filter and audio selection
            
        dims: dict,
            dictionary of dimensions to apply the transforms

    """

    def __init__(
        self,
        audio: torch.Tensor,
        sr: int,
        padding_length: int = 10000,
        deterministic: bool = False,
        dims={"ac": 0},
    ):
        self._audio = audio
        self._sr = sr
        self.padding_length = padding_length
        self.pad = torch.nn.ReflectionPad1d(self.padding_length)
        self.determinist = deterministic
        self.dims = dims
        self.dims.update({"all": slice(None)})

    @property
    def audio(self):
        """
        Get audio signal

        Returns:
            self._audio:torch.Tensor,
                audio of the class, with shape [dims,time_len]
        """
        return self._audio

    def audio(self, dim: str = "all"):
        """
        Get audio signal

        Returns:
            self._audio:torch.Tensor,
                audio of the class, with shape [dims,time_len]
        """
        return self._audio[self.dim(dim)]

    @property
    def sr(self):
        """
        Get sampling rate of the signal

        Returns:
            self._sr:int,
        """

        return self._sr

    def dim(self, dim: str = "all"):
        """
        Get the dimension index of the dims
        """
        if dim in self.dims:
            return self.dims[dim]
        raise ValueError(f"Dim {dim} is not found in dims {self.dims}.")

    def smoothing(
        self, smoothing_len: int = 512, dim: str = "all"
    ):
        """
        Smooth the signal's borders to get rid of any jump
        """
        dim = self.dim(dim)
        self._audio = self._audio.float()
        self._audio[dim, :smoothing_len] *= torch.pow(
            torch.linspace(0, 1, smoothing_len), 2
        )
        self._audio[dim, -smoothing_len:] *= torch.pow(
            torch.linspace(1, 0, smoothing_len), 2
        )

    def add_gauss_noise(
        self, intensity: float = 0.005, dim: str = "all"
    ):
        """
        Add gaussian noise to the signal

            Args:
                intensity: float,
                    intensity of the noise
                dims:Union[str, int, slice, list],
                
        """
        dim = self.dim(dim)
        mean = torch.zeros(size=self._audio[dim].shape)
        std = (
            intensity
            * self._audio[dim].std()
            * torch.ones(size=self._audio[dim].shape)
        )
        self._audio[dim] += torch.normal(mean=mean, std=std)

    def add_noise(
        self,
        noise: torch.Tensor,
        snr: float,
        dim: str = "all",
        inplace: bool = False,
    ):
        """
        Add noise to the signal with a given SNR. If noise is shorter than audio,
        it will be circularly repeated until long enough, then randomly sliced.

        Args:
            noise: torch.Tensor, shape [1, T_noise]
                Noise waveform to add.
            snr: float
                Desired signal-to-noise ratio (in dB).
            dim: str
                Dimension specifier (default = "all").
            inplace: bool
                If True, modify audio in-place. Otherwise, update self._audio[dim].
        """
        dim = self.dim(dim)
        audio_len = self._audio.shape[1]
        noise_len = noise.shape[1]

        if noise_len == 0:
            raise ValueError("Noise tensor is empty. Please provide valid noise data.")

        if noise_len < audio_len:
            repeat_times = (audio_len + noise_len - 1) // noise_len  
            noise = noise.repeat(1, repeat_times)
            noise_len = noise.shape[1]
        if self.determinist:
            start_idx = 0
        else:
            start_idx = torch.randint(0, noise_len - audio_len + 1, (1,))
        noise = noise[:, start_idx : start_idx + audio_len]

        snr_linear = 10 ** (snr / 10)
        signal_power = torch.mean(self._audio[dim] ** 2)
        noise_power = torch.mean(noise**2)

        if noise_power == 0:
            Warning("Noise power is zero, replacing with ε to avoid division by zero.")
            noise_power = 1e-6

        noise = torch.sqrt(signal_power / (noise_power * snr_linear)) * noise

        if inplace:
            self._audio[dim] += noise
        else:
            self._audio[dim] = self._audio[dim] + noise

    def remove_hf(
        self,
        cutoff_freq: int = 600,
        q_factor: float = 1,
        dim: str = "all",
    ):
        """
        Low-pass filter of the fourth order

            Args:

                cutoff_freq:int,
                    cutoff frequency of the filter

                Q:float,
                    Quality factor of the filter
        """
        dim = self.dim(dim)
        # pad for IRR response stabilisation
        self._audio = self.pad(self._audio).float()
        # filt-filt trick for 0-phase shift
        if not self.determinist:
            rand_factors = torch.FloatTensor(2).uniform_(0.8, 1.2)
        else:
            rand_factors = torch.FloatTensor(2).fill_(1.0)
        def lp(x):
            return lowpass_biquad(
                x,
                sample_rate=self._sr,
                cutoff_freq=cutoff_freq * rand_factors[0],
                Q=q_factor * rand_factors[1],
            )

        def reverse(x):
            return torch.flip(input=x, dims=[1])

        self._audio[dim] = reverse(lp(reverse(lp(self._audio[dim]))))
        self._audio = self._audio[..., self.padding_length : -self.padding_length]

    def normalize(
        self,
        percent=1.0,
        dim: str = "all",
        inplace=False,
    ):
        """
        Map audio values to [-1,1] and cut extremes values

        Args:
            percent: float,
                the percentage of values than will be kept before linear mapping,
                others are assigned to max or min

        """
        dim = self.dim(dim)
        if percent<=0.0 or percent>1.0:
            raise ValueError("percent must be in (0,1].")
        # 确保 dims 是一个可以迭代的对象
        if isinstance(dim, int):
            channels = [dim]
        elif isinstance(dim, slice):
            channels = range(*dim.indices(self._audio.shape[0]))
        elif isinstance(dim, list):
            channels = dim
        for channel in channels:
            sorted_audio, _ = torch.sort(abs(self._audio[channel]))
            sorted_audio = sorted_audio.flatten()

            # 确保 sorted_audio 不为空
            if sorted_audio.numel() == 0:
                continue

            # values of sorted audio
            cut = int(torch.numel(self._audio[channel]) * percent)
            new_abs_max = sorted_audio[cut - 1]

            self._audio[channel][self._audio[channel] > new_abs_max] = new_abs_max
            self._audio[channel][self._audio[channel] < -new_abs_max] = -new_abs_max

            a = 1 / new_abs_max if new_abs_max != 0 else 0
            if inplace:
                self._audio[channel] *= a
            else:
                self._audio[channel] = self._audio[channel] * a

    def resampling(self, new_freq: int):
        """
        Resample the signal to new_freq

            Args:
                new_freq:int,
                    new_freq of the signal
        """
        resampling = Resample(
            orig_freq=self._sr,
            resampling_method="sinc_interp_kaiser",
            new_freq=new_freq,
        )
        self._audio = resampling(self._audio)
        self._sr = new_freq

    def select_part(self, len_seconds: float):
        """
        Select a part of a signal

        Args:
            len_seconds: duration of selected signal

        """
        real_time_len = self._audio.shape[1]
        desired_time_len = int(len_seconds * self._sr)
        if real_time_len >= desired_time_len:  # cut tensor
            if not self.determinist:
                start_idx = torch.randint(
                    low=0, high=real_time_len - desired_time_len + 1, size=(1,)
                )
            else:
                start_idx = int((real_time_len - desired_time_len) / 4)
            self._audio = self._audio[:, start_idx : start_idx + desired_time_len]
        else:
            self._audio = torch.nn.functional.pad(
                input=self._audio,
                pad=(0, desired_time_len - real_time_len),
                mode="constant",
                value=0,
            )
