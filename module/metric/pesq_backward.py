# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch
from torch import Tensor, tensor

from torchmetrics.metric import Metric
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.imports import _MULTIPROCESSING_AVAILABLE, _PESQ_AVAILABLE
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _PESQ_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from typing import Any, Optional, Sequence, Union

__doctest_requires__ = {"PerceptualEvaluationSpeechQuality": ["pesq"]}
__doctest_requires__ = {("perceptual_evaluation_speech_quality",): ["pesq"]}
if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["PerceptualEvaluationSpeechQuality.plot"]
try:
    import pesq as pesq_backend
except:
    pesq_backend = None
    print("pesq_backend not found, please install pesq if you want to use this metric")

def perceptual_evaluation_speech_quality(
    preds: Tensor,
    target: Tensor,
    fs: int,
    mode: str,
    keep_same_device: bool = False,
    n_processes: int = 1,
) -> Tensor:
    r"""Calculate `Perceptual Evaluation of Speech Quality`_ (PESQ).

    It's a recognized industry standard for audio quality that takes into considerations characteristics such as: audio
    sharpness, call volume, background noise, clipping, audio interference etc. PESQ returns a score between -0.5 and
    4.5 with the higher scores indicating a better quality.

    This metric is a wrapper for the `pesq package`_. Note that input will be moved to `cpu` to perform the metric
    calculation.

    .. note:: using this metrics requires you to have ``pesq`` install. Either install as ``pip install
        torchmetrics[audio]`` or ``pip install pesq``. Note that ``pesq`` will compile with your currently
        installed version of numpy, meaning that if you upgrade numpy at some point in the future you will
        most likely have to reinstall ``pesq``.

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``
        fs: sampling frequency, should be 16000 or 8000 (Hz)
        mode: ``'wb'`` (wide-band) or ``'nb'`` (narrow-band)
        keep_same_device: whether to move the pesq value to the device of preds
        n_processes: integer specifying the number of processes to run in parallel for the metric calculation.
            Only applies to batches of data and if ``multiprocessing`` package is installed.

    Returns:
        Float tensor with shape ``(...,)`` of PESQ values per sample

    Raises:
        ModuleNotFoundError:
            If ``pesq`` package is not installed
        ValueError:
            If ``fs`` is not either  ``8000`` or ``16000``
        ValueError:
            If ``mode`` is not either ``"wb"`` or ``"nb"``
        RuntimeError:
            If ``preds`` and ``target`` do not have the same shape

    Example:
        >>> from torch import randn
        >>> from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
        >>> g = torch.manual_seed(1)
        >>> preds = randn(8000)
        >>> target = randn(8000)
        >>> perceptual_evaluation_speech_quality(preds, target, 8000, 'nb')
        tensor(2.2076)
        >>> perceptual_evaluation_speech_quality(preds, target, 16000, 'wb')
        tensor(1.7359)

    """
    if not _PESQ_AVAILABLE:
        raise ModuleNotFoundError(
            "PESQ metric requires that pesq is installed."
            " Either install as `pip install torchmetrics[audio]` or `pip install pesq`."
        )
    if fs not in (8000, 16000):
        raise ValueError(
            f"Expected argument `fs` to either be 8000 or 16000 but got {fs}"
        )
    if mode not in ("wb", "nb"):
        raise ValueError(
            f"Expected argument `mode` to either be 'wb' or 'nb' but got {mode}"
        )
    _check_same_shape(preds, target)

    if preds.ndim == 1:
        pesq_val_np = pesq_backend.pesq(
            fs,
            target.detach().cpu().numpy(),
            preds.detach().cpu().numpy(),
            mode,
            on_error=pesq_backend.PesqError.RETURN_VALUES,
        )
        pesq_val = torch.tensor(pesq_val_np)
    else:
        preds_np = preds.reshape(-1, preds.shape[-1]).detach().cpu().numpy()
        target_np = target.reshape(-1, preds.shape[-1]).detach().cpu().numpy()

        if _MULTIPROCESSING_AVAILABLE and n_processes != 1:
            pesq_val_np = pesq_backend.pesq_batch(
                fs,
                target_np,
                preds_np,
                mode,
                n_processor=n_processes,
                on_error=pesq_backend.PesqError.RETURN_VALUES,
            )
            pesq_val_np = np.array(pesq_val_np)
        else:
            pesq_val_np = np.empty(shape=(preds_np.shape[0]))
            for b in range(preds_np.shape[0]):
                pesq_val_np[b] = pesq_backend.pesq(
                    fs,
                    target_np[b, :],
                    preds_np[b, :],
                    mode,
                    on_error=pesq_backend.PesqError.RETURN_VALUES,
                )
        pesq_val = torch.from_numpy(pesq_val_np)
        pesq_val = pesq_val.reshape(preds.shape[:-1])

    if keep_same_device:
        return pesq_val.to(preds.device)

    return pesq_val

# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class PerceptualEvaluationSpeechQuality(Metric):
    """Calculate `Perceptual Evaluation of Speech Quality`_ (PESQ).

    It's a recognized industry standard for audio quality that takes into considerations characteristics such as:
    audio sharpness, call volume, background noise, clipping, audio interference etc. PESQ returns a score between
    -0.5 and 4.5 with the higher scores indicating a better quality.

    This metric is a wrapper for the `pesq package`_. Note that input will be moved to ``cpu`` to perform the metric
    calculation.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): float tensor with shape ``(...,time)``
    - ``target`` (:class:`~torch.Tensor`): float tensor with shape ``(...,time)``

    As output of `forward` and `compute` the metric returns the following output

    - ``pesq`` (:class:`~torch.Tensor`): float tensor of PESQ value reduced across the batch

    .. note:: using this metrics requires you to have ``pesq`` install. Either install as ``pip install
        torchmetrics[audio]`` or ``pip install pesq``. ``pesq`` will compile with your currently
        installed version of numpy, meaning that if you upgrade numpy at some point in the future you will
        most likely have to reinstall ``pesq``.

    .. note:: the ``forward`` and ``compute`` methods in this class return a single (reduced) PESQ value
        for a batch. To obtain a PESQ value for each sample, you may use the functional counterpart in
        :func:`~torchmetrics.functional.audio.pesq.perceptual_evaluation_speech_quality`.

    Args:
        fs: sampling frequency, should be 16000 or 8000 (Hz)
        mode: ``'wb'`` (wide-band) or ``'nb'`` (narrow-band)
        keep_same_device: whether to move the pesq value to the device of preds
        n_processes: integer specifying the number of processes to run in parallel for the metric calculation.
            Only applies to batches of data and if ``multiprocessing`` package is installed.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If ``pesq`` package is not installed
        ValueError:
            If ``fs`` is not either  ``8000`` or ``16000``
        ValueError:
            If ``mode`` is not either ``"wb"`` or ``"nb"``

    Example:
        >>> import torch
        >>> from torchmetrics.audio import PerceptualEvaluationSpeechQuality
        >>> g = torch.manual_seed(1)
        >>> preds = torch.randn(8000)
        >>> target = torch.randn(8000)
        >>> pesq = PerceptualEvaluationSpeechQuality(8000, 'nb')
        >>> pesq(preds, target)
        tensor(2.2076)
        >>> wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
        >>> wb_pesq(preds, target)
        tensor(1.7359)

    """

    sum_pesq: Tensor
    total: Tensor
    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = True
    plot_lower_bound: float = -0.5
    plot_upper_bound: float = 4.5

    def __init__(
        self,
        fs: int,
        mode: str,
        n_processes: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not _PESQ_AVAILABLE:
            raise ModuleNotFoundError(
                "PerceptualEvaluationSpeechQuality metric requires that `pesq` is installed."
                " Either install as `pip install torchmetrics[audio]` or `pip install pesq`."
            )
        if fs not in (8000, 16000):
            raise ValueError(
                f"Expected argument `fs` to either be 8000 or 16000 but got {fs}"
            )
        self.fs = fs
        if mode not in ("wb", "nb"):
            raise ValueError(
                f"Expected argument `mode` to either be 'wb' or 'nb' but got {mode}"
            )
        self.mode = mode
        if not isinstance(n_processes, int) and n_processes <= 0:
            raise ValueError(
                f"Expected argument `n_processes` to be an int larger than 0 but got {n_processes}"
            )
        self.n_processes = n_processes

        self.add_state("sum_pesq", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        pesq_batch = perceptual_evaluation_speech_quality(
            preds, target, self.fs, self.mode, False, self.n_processes
        ).to(self.sum_pesq.device)

        self.sum_pesq += pesq_batch.sum()
        self.total += pesq_batch.numel()

    def compute(self) -> Tensor:
        """Compute metric."""
        return self.sum_pesq / self.total

    def plot(
        self,
        val: Union[Tensor, Sequence[Tensor], None] = None,
        ax: Optional[_AX_TYPE] = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.audio import PerceptualEvaluationSpeechQuality
            >>> metric = PerceptualEvaluationSpeechQuality(8000, 'nb')
            >>> metric.update(torch.rand(8000), torch.rand(8000))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.audio import PerceptualEvaluationSpeechQuality
            >>> metric = PerceptualEvaluationSpeechQuality(8000, 'nb')
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.rand(8000), torch.rand(8000)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)

if __name__ == "__main__":
    preds = torch.randn(1, 16000)
    target = torch.randn(1, 16000)
    nb_pesq = PerceptualEvaluationSpeechQuality(16000, 'nb')
    nb_pesq.update(preds, target)
    print("nb_pesq: ", nb_pesq.compute())
    wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
    wb_pesq.update(preds, target)
    print("wb_pesq: ", wb_pesq.compute())
