import torch
import torch.nn as nn
import torch.nn.functional as F
from .sConformer import sConformer
import pytorch_lightning as pl


def match_shapes(encoder_output, decoder_output):
    """
    Adjust decoder enhanced to match encoder enhanced shape by cropping or padding.

    Args:
        encoder_output (torch.Tensor): Tensor with shape (B, C, H_enc, W_enc)
        decoder_output (torch.Tensor): Tensor with shape (B, C, H_dec, W_dec)

    Returns:
        torch.Tensor: Adjusted decoder enhanced
    """
    _, _, h_enc, w_enc = encoder_output.shape
    _, _, h_dec, w_dec = decoder_output.shape

    # Calculate differences
    pad_h = h_enc - h_dec  # Difference in height
    pad_w = w_enc - w_dec  # Difference in width

    # Cropping if decoder > encoder
    if pad_h < 0 or pad_w < 0:
        crop_h = max(0, -pad_h)
        crop_w = max(0, -pad_w)
        # Center cropping
        decoder_output = decoder_output[
            :, :, crop_h // 2 : crop_h // 2 + h_enc, crop_w // 2 : crop_w // 2 + w_enc
        ]

    # Padding if decoder < encoder
    if pad_h > 0 or pad_w > 0:
        padding = (0, max(0, pad_w), 0, max(0, pad_h))  # Pad left and right  # Pad top and bottom
        decoder_output = F.pad(decoder_output, padding, mode="constant", value=0)

    return decoder_output


# ----- STFT Processing -----
class STFT(nn.Module):
    def __init__(self, n_fft=320, hop_length=160, win_length=320):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)

    def forward(self, x):
        return torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device),
            return_complex=True,
        )

    def inverse(self, x):
        return torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device),
        ).unsqueeze(1)


# ----- Attention Modules -----
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionModule, self).__init__()
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(1, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
        )
        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(1, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        local_attn = self.local_att(x)
        global_attn = self.global_att(x)
        return self.sigmoid(local_attn + global_attn)


# ----- Iterative Attention Fusion -----
class IterativeAttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(IterativeAttentionFusion, self).__init__()
        self.attn1 = ChannelAttentionModule(in_channels)
        self.attn2 = ChannelAttentionModule(in_channels)

    def forward(self, y_ac, y_bc):
        """
        Args:
            y_ac: Air-conducted speech spectrogram (B, C, T, F)
            y_bc: Bone-conducted speech spectrogram (B, C, T, F)
        Returns:
            Fused spectrogram
        """
        # Coarse Fusion
        alpha = self.attn1(y_ac + y_bc)
        y_coarse = alpha * y_ac + (1 - alpha) * y_bc

        # Refined Fusion
        beta = self.attn2(y_coarse)
        y_refined = beta * y_ac + (1 - beta) * y_bc

        return y_refined


# ----- Dense Block -----
class EncoderDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate=8, num_layers=4):
        super(EncoderDenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels + i * growth_rate,
                        growth_rate,
                        kernel_size=(1, 3),
                        stride=(1, 1),
                        padding="same",
                    ),
                    nn.BatchNorm2d(growth_rate),
                    nn.PReLU(),
                )
            )
        self.conv = nn.Conv2d(
            in_channels=in_channels + num_layers * growth_rate,
            out_channels=out_channels,
            kernel_size=(1, 4),
            stride=(1, 2),
        )
        self.gated = nn.Conv2d(
            in_channels=in_channels + num_layers * growth_rate,
            out_channels=out_channels,
            kernel_size=(1, 4),
            stride=(1, 2),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        outputs = [x]
        for layer in self.layers:
            new_out = layer(torch.cat(outputs, dim=1))
            outputs.append(new_out)
        enhanced = torch.cat(outputs, dim=1)
        signal = self.conv(enhanced)
        gated = self.gated(enhanced)
        enhanced = signal * torch.sigmoid(gated)
        enhanced = self.prelu(self.bn(enhanced))
        return enhanced

class DecoderDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate=8, num_layers=4):
        super(DecoderDenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels + i * growth_rate,
                        growth_rate,
                        kernel_size=(1, 3),
                        stride=(1, 1),
                        padding="same",
                    ),
                    nn.BatchNorm2d(growth_rate),
                    nn.PReLU(),
                )
            )
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels + num_layers * growth_rate,
            out_channels=out_channels,
            kernel_size=(1, 4),
            stride=(1, 2),
            output_padding=(0, 1),
        )
        self.gated = nn.ConvTranspose2d(
            in_channels=in_channels + num_layers * growth_rate,
            out_channels=out_channels,
            kernel_size=(1, 4),
            stride=(1, 2),
            output_padding=(0, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        outputs = [x]
        for layer in self.layers:
            new_out = layer(torch.cat(outputs, dim=1))
            outputs.append(new_out)
        enhanced = torch.cat(outputs, dim=1)
        signal = self.conv(enhanced)
        gated = self.gated(enhanced)
        enhanced = signal * torch.sigmoid(gated)
        enhanced = self.prelu(self.bn(enhanced))
        return enhanced


# ----- AG Skip Connection -----
class AGSkipConnection(nn.Module):

    def __init__(self, channels):
        super(AGSkipConnection, self).__init__()
        self.conv_local = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_global = nn.Conv2d(1, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.pwconv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, encoder, decoder):
        decoder = match_shapes(encoder, decoder)
        local = self.conv_local(encoder + decoder)
        global_mean = torch.mean(encoder + decoder, dim=1, keepdim=True)
        global_attn = self.conv_global(global_mean)
        attn_coeff = self.sigmoid(local + global_attn)
        encoder_attn = attn_coeff * encoder
        encoder_attn = self.pwconv(encoder_attn)
        return torch.cat([encoder_attn, decoder], dim=1)


# ----- Full Model -----
class DenGCAN(pl.LightningModule):
    def __init__(
        self, 
        weight_decay=1e-4,
        metrics=None,
        train_metrics=None,
        val_metrics=None,
        test_metrics=None,
    ):
        super(DenGCAN, self).__init__()
        self.stft = STFT()
        self.fusion = IterativeAttentionFusion(in_channels=2)
        input_channels = 6
        encoder_channels = [16, 32, 48, 64, 64]
        decoder_channels = [64, 48, 32, 16, 2]

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        # Encoder Dense Blocks
        for out_channels in encoder_channels:
            self.encoder_blocks.append(
                EncoderDenseBlock(input_channels, out_channels, growth_rate=8, num_layers=5)
            )
            self.skip_connections.append(AGSkipConnection(out_channels))
            input_channels = out_channels

        # sConformer Bottleneck
        self.sconformer = sConformer(
            input_dim=192,
            num_heads=4,
            ffn_dim=128,
            num_layers=2,
            depthwise_conv_kernel_size=31,
            dropout=0.1,
        )

        # Decoder Dense Blocks
        for out_channels in decoder_channels:
            self.decoder_blocks.append(
                DecoderDenseBlock(input_channels * 2, out_channels, growth_rate=8, num_layers=5)
            )
            input_channels = out_channels

        self.weight_decay = weight_decay
        self.warmup_epochs = 10
        self.peak_lr = 2e-4
        self.final_lr = 2e-5

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

    def forward(self, ac, bc):
        # STFT: Convert input signals to spectrograms
        y_ac_stft = torch.view_as_real(self.stft(ac)).permute(0, 3, 2, 1)  # (B, 2, T, F)
        y_bc_stft = torch.view_as_real(self.stft(bc)).permute(0, 3, 2, 1)
        # Fusion
        fused = self.fusion(y_ac_stft, y_bc_stft)
        x = torch.cat([fused, y_ac_stft, y_bc_stft], dim=1)

        # DenGCAN Backbone

        skips = []
        for block in self.encoder_blocks:
            x = block(x)
            skips.append(x)

        b, c, t, f = x.shape
        x = x.transpose(1, 2).reshape([b, t, -1])
        x, _ = self.sconformer(x, torch.ones(x.shape[0], device=x.device) * x.shape[1])
        x = x.reshape([b, t, c, f]).transpose(1, 2)
        # Decoder Pass with Skip Connections
        for i, block in enumerate(self.decoder_blocks):
            x = self.skip_connections[-(i + 1)](skips[-(i + 1)], x)
            x = block(x)
        enhanced_signal = x * y_ac_stft
        enhanced_signal = torch.view_as_complex(enhanced_signal.permute(0, 3, 2, 1).contiguous())
        enhanced = self.stft.inverse(enhanced_signal)
        return enhanced, enhanced_signal

    def compute_loss(self, enhanced, reference):
        # Calculate L_RI
        real_diff = torch.abs(enhanced.real - reference.real)
        imag_diff = torch.abs(enhanced.imag - reference.imag)
        l_ri = torch.mean(real_diff + imag_diff)

        # Calculate L_Mag
        mag_diff = torch.abs(torch.abs(enhanced) - torch.abs(reference))
        l_mag = torch.mean(mag_diff)

        # Total loss
        loss = 0.5*l_ri + 0.5*l_mag
        return loss

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, eta_min=self.final_lr
            ),
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

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


# ----- Testing -----
if __name__ == "__main__":
    # Dummy input signals
    y_ac = torch.randn(4, 16000)  # Batch of air-conducted speech signals
    y_bc = torch.randn(4, 16000)  # Batch of bone-conducted speech signals

    # Instantiate model
    model = DenGCAN()

    # Forward pass
    enhanced = model(y_ac, y_bc)
    print("enhanced waveform shape:", enhanced.shape)
