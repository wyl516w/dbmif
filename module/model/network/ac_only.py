"""AC-only model support."""

import pytorch_lightning as pl
import torch


class OnlyAC(pl.LightningModule):
    """Single-input AC enhancement model."""

    def __init__(
        self,
        generator,
        discriminator=None,
        lr=None,
        betas=None,
        metrics=None,
        train_metrics=None,
        val_metrics=None,
        test_metrics=None,
        *args,
        **kwargs,
    ):
        super().__init__()

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

    def _prepare_batch(self, batch):
        cut_batch = [self.generator.cut_tensor(speech) for speech in batch[:3]]
        reference_speech, ac_speech, _ = cut_batch
        return reference_speech, ac_speech

    def forward(self, ac, bc):
        return self.generator(ac)

    def training_step(self, batch, batch_idx):
        reference_speech, corrupted_speech = self._prepare_batch(batch)
        opt_g, opt_d = self.optimizers()

        enhanced_speech, decomposed_enhanced_speech = self.generator(corrupted_speech)
        decomposed_reference_speech = self.generator.pqmf.forward(
            reference_speech, "analysis"
        )
        enhanced_embeddings = self.discriminator(
            bands=decomposed_enhanced_speech[:, -self.discriminator.q :, :],
            audio=enhanced_speech,
        )
        reference_embeddings = self.discriminator(
            bands=decomposed_reference_speech[:, -self.discriminator.q :, :],
            audio=reference_speech,
        )

        ftr_loss = 0
        for scale in range(len(reference_embeddings)):
            for layer in range(1, len(reference_embeddings[scale]) - 1):
                a = reference_embeddings[scale][layer]
                b = enhanced_embeddings[scale][layer]
                ftr_loss += self.l1(a, b) / (len(reference_embeddings[scale]) - 2)
        ftr_loss /= len(reference_embeddings)

        adv_loss = 0
        for scale in range(len(enhanced_embeddings)):
            certainties = enhanced_embeddings[scale][-1]
            adv_loss += self.relu(1 - certainties).mean()
        adv_loss /= len(enhanced_embeddings)

        gen_loss = adv_loss + 100 * ftr_loss
        opt_g.zero_grad()
        self.manual_backward(gen_loss)
        opt_g.step()

        enhanced_speech, decomposed_enhanced_speech = self.generator(corrupted_speech)
        decomposed_reference_speech = self.generator.pqmf.forward(
            reference_speech, "analysis"
        )
        enhanced_embeddings = self.discriminator(
            bands=decomposed_enhanced_speech[:, -self.discriminator.q :, :],
            audio=enhanced_speech,
        )
        reference_embeddings = self.discriminator(
            bands=decomposed_reference_speech[:, -self.discriminator.q :, :],
            audio=reference_speech,
        )

        adv_loss_valid = 0
        for scale in range(len(reference_embeddings)):
            certainties = reference_embeddings[scale][-1]
            adv_loss_valid += self.relu(1 - certainties).mean()
        adv_loss_valid /= len(reference_embeddings)

        adv_loss_fake = 0
        for scale in range(len(enhanced_embeddings)):
            certainties = enhanced_embeddings[scale][-1]
            adv_loss_fake += self.relu(1 + certainties).mean()
        adv_loss_fake /= len(enhanced_embeddings)

        dis_loss = adv_loss_valid + adv_loss_fake
        opt_d.zero_grad()
        self.manual_backward(dis_loss)
        opt_d.step()

        outs = {
            "reference": reference_speech,
            "corrupted": corrupted_speech,
            "enhanced": enhanced_speech,
            "reference_embeddings": reference_embeddings,
            "enhanced_embeddings": enhanced_embeddings,
            "loss_dict": {
                "discriminator_valid_loss": adv_loss_valid,
                "discriminator_fake_loss": adv_loss_fake,
                "discriminator_loss": dis_loss,
                "generator_ftr_loss": ftr_loss,
                "generator_adv_loss": adv_loss,
                "generator_loss": gen_loss,
                "total_loss": dis_loss + gen_loss,
            },
        }
        channel_losses = []
        for channel in range(decomposed_enhanced_speech.shape[1]):
            loss = self.l1(
                decomposed_enhanced_speech[:, channel, :],
                decomposed_reference_speech[:, channel, :],
            )
            channel_losses.append(loss)
        for index, loss in enumerate(channel_losses):
            outs["loss_dict"][f"channel_{index}_loss"] = loss
        return outs

    def validation_step(self, batch, batch_idx):
        reference_speech, corrupted_speech = self._prepare_batch(batch)
        enhanced_speech, _ = self.generator(corrupted_speech)
        return {
            "reference": reference_speech,
            "corrupted": corrupted_speech,
            "enhanced": enhanced_speech,
        }

    def test_step(self, batch, batch_idx):
        reference_speech, corrupted_speech = self._prepare_batch(batch)
        enhanced_speech, _ = self.generator(corrupted_speech)
        return {
            "reference": reference_speech,
            "corrupted": corrupted_speech,
            "enhanced": enhanced_speech,
        }

    def predict_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            corrupted_speech = batch[0]
        else:
            corrupted_speech = batch
        corrupted_speech = self.generator.cut_tensor(corrupted_speech)
        enhanced_speech, _ = self.generator(corrupted_speech)
        return {
            "corrupted": corrupted_speech,
            "enhanced": enhanced_speech,
        }

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            params=self.generator.parameters(), lr=self.lr, betas=self.betas
        )
        opt_d = torch.optim.Adam(
            params=self.discriminator.parameters(), lr=self.lr, betas=self.betas
        )
        return opt_g, opt_d

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
        origin_metricsdict = metrics_collection(
            outputs["corrupted"], outputs["reference"]
        )
        self.log_dict(
            {k + "/origin": v for k, v in origin_metricsdict.items()},
            on_epoch=on_epoch,
            on_step=on_step,
            sync_dist=sync_dist,
        )
        enhanced_metricsdict = metrics_collection(
            outputs["enhanced"], outputs["reference"]
        )
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
