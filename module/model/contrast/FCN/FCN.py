# FCN models for ACM and BCM signals
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # FCNEF is composed of seven hidden convolutional layers, each layer contains 30 kernels of size 55. In addition, to examine the impact of BCM, we construct another FCN model that is close to the FCNEF while it only adopts the ACM channel. Evaluations for these models will be described in Section IV.
        self.conv1 = nn.Conv1d(2, 64, kernel_size=55, padding="same")
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=55, padding="same")
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=55, padding="same")
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=55, padding="same")
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 128, kernel_size=55, padding="same")
        self.bn5 = nn.BatchNorm1d(128)
        self.conv6 = nn.Conv1d(128, 64, kernel_size=55, padding="same")
        self.bn6 = nn.BatchNorm1d(64)
        self.conv7 = nn.Conv1d(64, 32, kernel_size=55, padding="same")
        self.bn7 = nn.BatchNorm1d(32)
        self.conv8 = nn.Conv1d(32, 1, kernel_size=55, padding="same")
        
    def forward(self, input_bone, input_air):
        x = torch.cat((input_bone, input_air), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.conv8(x)
        return torch.tanh(x)


import pytorch_lightning as pl


class FCN(pl.LightningModule):

    def __init__(
        self,
        metrics=None,
        train_metrics=None,
        val_metrics=None,
        test_metrics=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.fcn = ConvNet()
        self.loss = nn.MSELoss()
        if metrics:
            self.train_metrics = metrics.clone(prefix="train/")
            self.val_metrics = metrics.clone(prefix="val/")
            self.test_metrics = metrics.clone(prefix="test/")
        if train_metrics:
            self.train_metrics = train_metrics.clone(prefix="train/")
        if val_metrics:
            self.val_metrics = val_metrics.clone(prefix="val/")
        if test_metrics:
            self.test_metrics = test_metrics.clone(prefix="test/")

    def forward(self, acm, bcm):
        return self.fcn(acm, bcm)

    def training_step(self, batch, batch_idx):
        target, acm, bcm = batch
        output = self(acm, bcm)
        loss = self.loss(output, target)
        self.log("train_loss", loss)
        return {"loss": loss, "output": output, "target": target, "acm": acm, "bcm": bcm}

    def validation_step(self, batch, batch_idx):
        target, acm, bcm = batch
        output = self(acm, bcm)
        loss = self.loss(output, target)
        self.log("val_loss", loss)
        return {"loss": loss, "output": output, "target": target, "acm": acm, "bcm": bcm}

    def test_step(self, batch, batch_idx):
        target, acm, bcm = batch
        output = self(acm, bcm)
        loss = self.loss(output, target)
        self.log("test_loss", loss)
        return {"loss": loss, "output": output, "target": target, "acm": acm, "bcm": bcm}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def logging(
        self,
        outputs,
        metrics_collection,
        prefix="all",
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    ):
        if outputs.get("loss_dict"):
            for metricname, metricvalue in outputs["loss_dict"].items():
                self.log(
                    prefix + "/" + metricname,
                    metricvalue,
                    on_epoch=False,
                    on_step=True,
                    sync_dist=sync_dist,
                )
        origin_ac_metricsdict = metrics_collection(outputs["acm"], outputs["target"])
        self.log_dict(
            {k + "/origin_ac": v for k, v in origin_ac_metricsdict.items()},
            on_epoch=on_epoch,
            on_step=on_step,
            sync_dist=sync_dist,
        )
        origin_bc_metricsdict = metrics_collection(outputs["bcm"], outputs["target"])
        self.log_dict(
            {k + "/origin_bc": v for k, v in origin_bc_metricsdict.items()},
            on_epoch=on_epoch,
            on_step=on_step,
            sync_dist=sync_dist,
        )
        enhanced_metricsdict = metrics_collection(outputs["output"], outputs["target"])
        self.log_dict(
            {k + "/enhanced": v for k, v in enhanced_metricsdict.items()},
            on_epoch=on_epoch,
            on_step=on_step,
            sync_dist=sync_dist,
        )

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        self.logging(
            outputs,
            self.train_metrics,
            prefix="train",
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

    def on_validation_batch_end(self, outputs, batch, batch_idx: int) -> None:
        self.logging(
            outputs,
            self.val_metrics,
            prefix="val",
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

    def on_test_batch_end(self, outputs, batch, batch_idx: int) -> None:
        self.logging(
            outputs,
            self.test_metrics,
            prefix="test",
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
