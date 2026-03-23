import torchmetrics
from torch import Tensor


class MeanSquaredError(torchmetrics.MeanSquaredError):
    def update(self, preds: Tensor, target: Tensor) -> None:
        target = target.reshape(-1)
        preds = preds.reshape(-1)
        super().update(preds, target)

if __name__ == "__main__":
    import torch
    preds = torch.randn(1, 16000)
    target = torch.randn(1, 16000)
    mse = MeanSquaredError()
    mse.update(preds, target)
    print(mse.compute())
