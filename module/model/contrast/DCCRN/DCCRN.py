import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import pytorch_lightning as pl


class TransposeDenseBlock(nn.Module):
    def __init__(self, input_channels, output_channels, growth_rate=8, layers=5):
        super(TransposeDenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.ZeroPad2d((0, 0, 1, 0)),
                    nn.Conv2d(
                        input_channels + i * growth_rate,
                        growth_rate,
                        kernel_size=(4, 1),
                        padding=(1, 0),
                    ),
                    nn.BatchNorm2d(growth_rate),
                    nn.PReLU(),
                )
            )
        self.conv1 = nn.ConvTranspose2d(
            input_channels + (layers - 1) * growth_rate,
            output_channels,
            kernel_size=(5, 3),
            stride=(2, 1),
            padding=(2, 1),
        )
        self.conv2 = nn.ConvTranspose2d(
            input_channels + (layers - 1) * growth_rate,
            output_channels,
            kernel_size=(5, 3),
            stride=(2, 1),
            padding=(2, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        x_cat = torch.cat(features, dim=1)
        gated_feature = self.conv1(x_cat) * self.sigmoid(self.conv2(x_cat))
        return gated_feature


class DenseBlock(nn.Module):
    def __init__(self, input_channels, output_channels, growth_rate=8, layers=5):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.ZeroPad2d((0, 0, 1, 0)),
                    nn.Conv2d(
                        input_channels + i * growth_rate,
                        growth_rate,
                        kernel_size=(4, 1),
                        padding=(1, 0),
                    ),
                    nn.BatchNorm2d(growth_rate),
                    nn.PReLU(),
                )
            )
        self.conv1 = nn.Conv2d(
            input_channels + (layers - 1) * growth_rate,
            output_channels,
            kernel_size=(5, 3),
            stride=(2, 1),
            padding=(2, 1),
        )
        self.conv2 = nn.Conv2d(
            input_channels + (layers - 1) * growth_rate,
            output_channels,
            kernel_size=(5, 3),
            stride=(2, 1),
            padding=(2, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        x_cat = torch.cat(features, dim=1)
        gated_feature = self.conv1(x_cat) * self.sigmoid(self.conv2(x_cat))
        return gated_feature


class AFF(nn.Module):
    def __init__(self, input_channels):
        super(AFF, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.local_conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, kernel_size=1),
            nn.BatchNorm2d(input_channels // 2),
            nn.PReLU(),
            nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=1),
            nn.BatchNorm2d(input_channels // 2),
            nn.PReLU(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, ac, bc):
        # Calculate attention score M
        global_context = self.global_avg_pool(ac + bc)
        local_context = self.local_conv(ac + bc)
        attention_score = self.sigmoid(global_context + local_context)
        # Perform soft selection and feature concatenation
        fused_ac = attention_score * ac
        fused_bc = (1 - attention_score) * bc
        fused = fused_ac + fused_bc
        return torch.cat([fused_ac, fused_bc, fused], dim=1)


class DCCRN(pl.LightningModule):
    def __init__(
        self,
        rnn_units=256,
        num_layers=2,
        n_fft=256,
        hop_length=128,
        win_length=256,
        casual=True,
        metrics=None,
        train_metrics=None,
        val_metrics=None,
        test_metrics=None,
    ):
        super(DCCRN, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        input_channels = 2
        self.loss = nn.MSELoss()

        # Define encoder, LSTM, and decoder components
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    DenseBlock(input_channels * 3, 16, growth_rate=8, layers=5),
                    nn.BatchNorm2d(16),
                    nn.PReLU(),
                ),
                nn.Sequential(
                    DenseBlock(16, 16, growth_rate=8, layers=5), nn.BatchNorm2d(16), nn.PReLU()
                ),
                nn.Sequential(
                    DenseBlock(16, 16, growth_rate=8, layers=5), nn.BatchNorm2d(16), nn.PReLU()
                ),
                nn.Sequential(
                    DenseBlock(16, 32, growth_rate=8, layers=5), nn.BatchNorm2d(32), nn.PReLU()
                ),
                nn.Sequential(
                    DenseBlock(32, 64, growth_rate=8, layers=5), nn.BatchNorm2d(64), nn.PReLU()
                ),
                nn.Sequential(
                    DenseBlock(64, 128, growth_rate=8, layers=5), nn.BatchNorm2d(128), nn.PReLU()
                ),
                nn.Sequential(
                    DenseBlock(128, 256, growth_rate=8, layers=5), nn.BatchNorm2d(256), nn.PReLU()
                ),
                # nn.Sequential(
                #     DenseBlock(256, 256, growth_rate=8, layers=5), nn.BatchNorm2d(256), nn.PReLU()
                # ),
                # nn.Sequential(
                #     DenseBlock(256, 256, growth_rate=8, layers=5), nn.BatchNorm2d(256), nn.PReLU()
                # ),
            ]
        )

        # self.bridge = nn.ModuleList(
        #     [
        #         nn.Conv2d(16, 16, kernel_size=1, stride=1),
        #         nn.Conv2d(32, 32, kernel_size=1, stride=1),
        #         nn.Conv2d(64, 64, kernel_size=1, stride=1),
        #         nn.Conv2d(128, 128, kernel_size=1, stride=1),
        #         nn.Conv2d(256, 256, kernel_size=1, stride=1),
        #         # nn.Conv2d(256, 256, kernel_size=1, stride=1),
        #         # nn.Conv2d(256, 256, kernel_size=1, stride=1),
        #     ]
        # )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=rnn_units*(2 if casual else 1),
            num_layers=num_layers,
            batch_first=True,
            bidirectional=not casual,
        )

        self.decoder = nn.ModuleList(
            [
                # nn.Sequential(
                #     TransposeDenseBlock(256, 256, growth_rate=8, layers=5),
                #     nn.BatchNorm2d(256),
                #     nn.PReLU(),
                # ),
                # nn.Sequential(
                #     TransposeDenseBlock(256, 256, growth_rate=8, layers=5),
                #     nn.BatchNorm2d(256),
                #     nn.PReLU(),
                # ),
                nn.Sequential(
                    TransposeDenseBlock(256, 128, growth_rate=8, layers=5),
                    nn.BatchNorm2d(128),
                    nn.PReLU(),
                ),
                nn.Sequential(
                    TransposeDenseBlock(128, 64, growth_rate=8, layers=5),
                    nn.BatchNorm2d(64),
                    nn.PReLU(),
                ),
                nn.Sequential(
                    TransposeDenseBlock(64, 32, growth_rate=8, layers=5),
                    nn.BatchNorm2d(32),
                    nn.PReLU(),
                ),
                nn.Sequential(
                    TransposeDenseBlock(32, 16, growth_rate=8, layers=5),
                    nn.BatchNorm2d(16),
                    nn.PReLU(),
                ),
                nn.Sequential(
                    TransposeDenseBlock(16, 16, growth_rate=8, layers=5),
                    nn.BatchNorm2d(16),
                    nn.PReLU(),
                ),
                nn.Sequential(
                    TransposeDenseBlock(16, 16, growth_rate=8, layers=5),
                    nn.BatchNorm2d(16),
                    nn.PReLU(),
                ),
                nn.Sequential(
                    TransposeDenseBlock(16, input_channels, growth_rate=8, layers=5),
                    nn.BatchNorm2d(input_channels),
                    nn.PReLU(),
                ),
            ]
        )

        self.attention_fusion = AFF(input_channels)
        self.conv_out = nn.Conv2d(input_channels, 2, kernel_size=1)

        # Metrics
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

    def forward(self, noisy_ac, bc):
        # STFT transformation
        ac_spec = self.stft(noisy_ac)
        bc_spec = self.stft(bc)
        ac_spec_real = torch.view_as_real(ac_spec).permute(0, 3, 1, 2)
        bc_spec_real = torch.view_as_real(bc_spec).permute(0, 3, 1, 2)

        # Attention-based fusion
        x = self.attention_fusion(ac_spec_real, bc_spec_real)
        # Encoder forward pass
        # skip_connections = []
        # bridge_outputs = []
        for idx, block in enumerate(self.encoder):
            x = block(x)
            # bridge_output = self.bridge[idx](x)
            # bridge_outputs.append(bridge_output)
            # skip_connections.append(x)

        # Reshape to fit LSTM input (batch_size, sequence_len, features)
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, time, -1)
        # LSTM forward pass
        x, _ = self.lstm(x)
        # Reshape back to 4D tensor (batch_size, channels, freq, time)
        x = x.view(batch_size, time, channels, freq).permute(0, 2, 3, 1)

        # Decoder forward pass
        for idx, block in enumerate(self.decoder):
            # x = x + bridge_outputs[-(idx + 1)]
            x = block(x)

        # enhanced layer to estimate real and imaginary parts of enhanced speech
        out = self.conv_out(x).permute(0, 2, 3, 1).contiguous()
        out = torch.view_as_complex(out)
        # Inverse STFT to obtain time-domain signal
        enhanced_signal = self.istft(out)
        return enhanced_signal, out

    def stft(self, x):
        window = torch.hann_window(self.win_length).to(x.device)
        return torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )

    def istft(self, x):
        window = torch.hann_window(self.win_length).to(x.device)
        return torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
        ).unsqueeze(1)

    def training_step(self, batch, batch_idx):
        reference, corrupted_ac, corrupted_bc = batch
        enhanced, signal = self(corrupted_ac, corrupted_bc)
        loss = self.compute_loss(signal, self.stft(reference))
        self.log("train_loss", loss)
        return {"loss": loss, "enhanced": enhanced, "reference": reference, "corrupted_ac": corrupted_ac, "corrupted_bc": corrupted_bc}

    def validation_step(self, batch, batch_idx):
        reference, corrupted_ac, corrupted_bc = batch
        enhanced, signal = self(corrupted_ac, corrupted_bc)
        loss = self.compute_loss(signal, self.stft(reference))
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return {"loss": loss, "enhanced": enhanced, "reference": reference, "corrupted_ac": corrupted_ac, "corrupted_bc": corrupted_bc}

    def test_step(self, batch, batch_idx):
        reference, corrupted_ac, corrupted_bc = batch
        enhanced, signal = self(corrupted_ac, corrupted_bc)
        loss = self.compute_loss(signal, self.stft(reference))
        self.log("test_loss", loss)
        return {"loss": loss, "enhanced": enhanced, "reference": reference, "corrupted_ac": corrupted_ac, "corrupted_bc": corrupted_bc}

    def compute_loss(self, enhanced, reference):
        # Calculate L_RI
        real_diff = torch.abs(enhanced.real - reference.real)
        imag_diff = torch.abs(enhanced.imag - reference.imag)
        l_ri = torch.mean(real_diff + imag_diff)

        # Calculate L_Mag
        mag_diff = torch.abs(torch.abs(enhanced) - torch.abs(reference))
        l_mag = torch.mean(mag_diff)

        # Total loss
        loss = l_ri + l_mag
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=6e-4)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3
            ),
            'interval': 'epoch',  # 调度器在每个 epoch 结束时调用
            'monitor': 'val_loss',  # 监控的指标
            'strict': False,  # 如果未找到监控指标，则停止训练
        }
        return {"optimizer":optimizer, "lr_scheduler": scheduler}
        # return optimizer

    def logging(
        self,
        outputs,
        metrics_collection,
        prefix="all",
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    ):
        # cut reference, corrupted_ac, corrupted_bc to enhanced
        outputs["reference"] = outputs["reference"][:, :, : outputs["enhanced"].shape[2]]
        outputs["corrupted_ac"] = outputs["corrupted_ac"][:, :, : outputs["enhanced"].shape[2]]
        outputs["corrupted_bc"] = outputs["corrupted_bc"][:, :, : outputs["enhanced"].shape[2]]

        if outputs.get("loss_dict"):
            for metricname, metricvalue in outputs["loss_dict"].items():
                self.log(
                    prefix + "/" + metricname,
                    metricvalue,
                    on_epoch=False,
                    on_step=True,
                    sync_dist=sync_dist,
                )
        origin_ac_metricsdict = metrics_collection(
            outputs["corrupted_ac"], outputs["reference"]
        )
        self.log_dict(
            {k + "/origin_ac": v for k, v in origin_ac_metricsdict.items()},
            on_epoch=on_epoch,
            on_step=on_step,
            sync_dist=sync_dist,
        )
        origin_bc_metricsdict = metrics_collection(
            outputs["corrupted_bc"], outputs["reference"]
        )
        self.log_dict(
            {k + "/origin_bc": v for k, v in origin_bc_metricsdict.items()},
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


if __name__ == "__main__":
    model = DCCRN()
    corrupted_ac = torch.randn(4, 1, 16000)
    corrupted_bc = torch.randn(4, 1, 16000)
    reference = torch.randn(4, 1, 16000)
    enhanced,_ = model(corrupted_ac, corrupted_bc)
    print(enhanced.shape)  # torch.Size([4, 1, 16000])
