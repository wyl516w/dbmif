"""Definition of ACBC and its training pipeline with pytorch lightning."""

from abc import abstractmethod
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


class ACBC(pl.LightningModule):
    """
    ACBC LightningModule
    """

    def __init__(
        self,
        generator,
        discriminator=None,
        lr=None,
        betas=None,
        scheduler=None,
        metrics=None,
        train_metrics=None,
        val_metrics=None,
        test_metrics=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.strict_loading = False
        if isinstance(generator, dict):
            from ...utils.utils import initialize_model_from_dict

            self.generator = initialize_model_from_dict(generator)
        else:
            self.generator = generator
        if isinstance(discriminator, dict):
            from ...utils.utils import initialize_model_from_dict

            self.discriminator = initialize_model_from_dict(discriminator)
        else:
            self.discriminator = discriminator
        self.lr = lr
        self.betas = tuple(betas)
        self.l1 = torch.nn.L1Loss()
        self.relu = torch.nn.ReLU()
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
        self.automatic_optimization = False
        self.origin_shape = None
        self.forward = self.generator.forward
        self.scheduler = scheduler

    def _generator_step_nograd(self, batch):
        if len(batch) == 2:
            corrupted_ac, corrupted_bc = batch
            reference = None
        else:
            reference, corrupted_ac, corrupted_bc = batch
        enhanced, _ = self.generator(corrupted_ac, corrupted_bc)
        return {
            "reference": reference,
            "corrupted_ac": corrupted_ac,
            "corrupted_bc": corrupted_bc,
            "enhanced": enhanced,
        }

    def _generator_step(self, batch):
        reference, corrupted_ac, corrupted_bc = batch
        enhanced, decomposed_enhanced = self.generator(corrupted_ac, corrupted_bc)
        decomposed_reference = self.generator.pqmf_analysis(reference)
        enhanced_embeddings = self.discriminator(
            bands=decomposed_enhanced[:, -self.discriminator.q :, :],
            audio=enhanced,
        )
        reference_embeddings = self.discriminator(
            bands=decomposed_reference[:, -self.discriminator.q :, :],
            audio=reference,
        )

        # train generator
        # ftr_loss
        ftr_loss = 0
        for scale in range(len(reference_embeddings)):  # across scales
            for layer in range(1, len(reference_embeddings[scale]) - 1):  # across layers
                a = reference_embeddings[scale][layer]
                b = enhanced_embeddings[scale][layer]
                ftr_loss += self.l1(a, b) / (len(reference_embeddings[scale]) - 2)
        ftr_loss /= len(reference_embeddings)
        # loss_adv_gen
        adv_loss = 0
        for scale in range(len(enhanced_embeddings)):  # across embeddings
            certainties = enhanced_embeddings[scale][-1]
            adv_loss += self.relu(1 - certainties).mean()  # across time
        adv_loss /= len(enhanced_embeddings)
        channels_loss_list = []
        for channel in range(decomposed_enhanced.shape[1]):
            loss = self.l1(
                decomposed_enhanced[:, channel, :],
                decomposed_reference[:, channel, :],
            )
            channels_loss_list.append(loss)
        gen_loss = adv_loss + 1000 * ftr_loss + 0.1 * F.l1_loss(enhanced, reference) + 0.1 * F.l1_loss(decomposed_enhanced, decomposed_reference)
        outs = {
            "reference": reference,
            "corrupted_ac": corrupted_ac,
            "corrupted_bc": corrupted_bc,
            "enhanced": enhanced,
            "reference_embeddings": reference_embeddings,
            "enhanced_embeddings": enhanced_embeddings,
            "loss_dict": {
                "generator_ftr_loss": ftr_loss,
                "generator_adv_loss": adv_loss,
                "generator_loss": gen_loss,
            },
        }

        if hasattr(self.generator, "external_loss_value"):
            ext_loss = self.generator.external_loss_value
            outs["loss_dict"].update({"external_loss": ext_loss})
        outs["loss_dict"].update({f"channel_{i}_loss": loss for i, loss in enumerate(channels_loss_list)})
        return outs

    def _discriminator_step(self, batch):
        reference, corrupted_ac, corrupted_bc = batch
        enhanced, decomposed_enhanced = self.generator(corrupted_ac, corrupted_bc)
        decomposed_reference = self.generator.pqmf.forward(reference, "analysis")
        enhanced_embeddings = self.discriminator(
            bands=decomposed_enhanced[:, -self.discriminator.q :, :],
            audio=enhanced,
        )
        reference_embeddings = self.discriminator(
            bands=decomposed_reference[:, -self.discriminator.q :, :],
            audio=reference,
        )

        # train discriminator
        # valid_loss
        adv_loss_valid = 0
        for scale in range(len(reference_embeddings)):  # across embeddings
            certainties = reference_embeddings[scale][-1]
            adv_loss_valid += self.relu(1 - certainties).mean()  # across time
        adv_loss_valid /= len(reference_embeddings)
        # fake_loss
        adv_loss_fake = 0
        for scale in range(len(enhanced_embeddings)):  # across embeddings
            certainties = enhanced_embeddings[scale][-1]
            adv_loss_fake += self.relu(1 + certainties).mean()  # across time
        adv_loss_fake /= len(enhanced_embeddings)
        # loss to backprop on
        dis_loss = adv_loss_valid + adv_loss_fake
        outs = {
            "discriminator_valid_loss": adv_loss_valid,
            "discriminator_fake_loss": adv_loss_fake,
            "discriminator_loss": dis_loss,
        }
        return outs

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        outs = self._generator_step(batch)
        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(outs["loss_dict"]["generator_loss"])
        opt_g.step()
        outs.update(self._discriminator_step(batch))
        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(outs["discriminator_loss"])
        opt_d.step()
        return outs

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if batch_idx % 2 == 0:
            self.outputs = outputs
        else:
            self.outputs.update(outputs)
            self.logging(
                self.outputs,
                self.train_metrics,
                prefix="train",
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

    def on_train_epoch_end(self) -> None:
        sch_g, sch_d = self.lr_schedulers()
        if isinstance(sch_g, CosineAnnealingLR):
            sch_g.step()
            self.log("lr/generator", sch_g.optimizer.param_groups[0]["lr"])
        if isinstance(sch_d, CosineAnnealingLR):
            sch_d.step()
            self.log("lr/discriminator", sch_d.optimizer.param_groups[0]["lr"])
        return super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        self.val_loss = 0
        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        outs = self._generator_step_nograd(batch)
        self.val_loss += F.mse_loss(outs["enhanced"], outs["reference"])
        return outs

    def on_validation_batch_end(self, outputs, batch, batch_idx: int) -> None:
        self.logging(
            outputs,
            self.val_metrics,
            prefix="val",
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:
        sch_g, sch_d = self.lr_schedulers()
        if isinstance(sch_g, ReduceLROnPlateau):
            self.log("lr/generator", sch_g.optimizer.param_groups[0]["lr"])
            sch_g.step(self.val_loss)
        if isinstance(sch_d, ReduceLROnPlateau):
            sch_d.step(self.val_loss)
            self.log("lr/discriminator", sch_d.optimizer.param_groups[0]["lr"])
        self.log("lr/loss", self.val_loss)
        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        outs = self._generator_step_nograd(batch)
        return outs

    def on_test_batch_end(self, outputs, batch, batch_idx: int) -> None:
        self.logging(
            outputs,
            self.test_metrics,
            prefix="test",
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

    def predict_step(self, batch, batch_idx):
        outs = self._generator_step_nograd(batch)
        return outs

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(params=self.generator.parameters(), lr=self.lr, betas=self.betas)
        opt_d = torch.optim.Adam(params=self.discriminator.parameters(), lr=self.lr, betas=self.betas)
        if self.scheduler == "cosine":
            scheduler_g = CosineAnnealingLR(optimizer=opt_g, T_max=800, eta_min=1e-6)
            scheduler_d = CosineAnnealingLR(optimizer=opt_d, T_max=800, eta_min=1e-6)
        elif self.scheduler == "plateau":
            scheduler_g = ReduceLROnPlateau(optimizer=opt_g, mode="min", factor=0.5, patience=3)
            scheduler_d = ReduceLROnPlateau(optimizer=opt_d, mode="min", factor=0.5, patience=3)
        else:
            scheduler_g = CosineAnnealingLR(optimizer=opt_g, T_max=800, eta_min=1e-6)
            scheduler_d = CosineAnnealingLR(optimizer=opt_g, T_max=800, eta_min=1e-6)
        return {"optimizer": opt_g, "lr_scheduler": scheduler_g}, {"optimizer": opt_d, "lr_scheduler": scheduler_d}

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
        origin_ac_metricsdict = metrics_collection(outputs["corrupted_ac"], outputs["reference"])
        self.log_dict(
            {k + "/origin_ac": v for k, v in origin_ac_metricsdict.items()},
            on_epoch=on_epoch,
            on_step=on_step,
            sync_dist=sync_dist,
        )
        origin_bc_metricsdict = metrics_collection(outputs["corrupted_bc"], outputs["reference"])
        self.log_dict(
            {k + "/origin_bc": v for k, v in origin_bc_metricsdict.items()},
            on_epoch=on_epoch,
            on_step=on_step,
            sync_dist=sync_dist,
        )
        enhanced_metricsdict = metrics_collection(outputs["enhanced"], outputs["reference"])
        self.log_dict(
            {k + "/enhanced": v for k, v in enhanced_metricsdict.items()},
            on_epoch=on_epoch,
            on_step=on_step,
            sync_dist=sync_dist,
        )
