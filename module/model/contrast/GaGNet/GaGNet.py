import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor


class GaGNet(nn.Module):
    def __init__(
        self,
        cin: int = 2,
        k1: tuple = (2, 3),
        k2: tuple = (1, 3),
        c: int = 64,
        kd1: int = 3,
        cd1: int = 64,
        d_feat: int = 256,
        p: int = 2,
        q: int = 3,
        dilas: list = [1, 2, 5, 9],
        fft_num: int = 320,
        is_u2: bool = True,
        is_causal: bool = True,
        is_squeezed: bool = False,
        acti_type: str = "sigmoid",
        intra_connect: str = "cat",
        norm_type: str = "IN",
    ):
        super().__init__()

        if is_u2:
            self.en = U2Net_Encoder(cin, k1, k2, c, intra_connect, norm_type)
        else:
            self.en = UNet_Encoder(cin, k1, c, norm_type)
        self.gags = nn.ModuleList(
            [
                GlanceGazeModule(
                    kd1,
                    cd1,
                    d_feat,
                    p,
                    dilas,
                    fft_num,
                    is_causal,
                    is_squeezed,
                    acti_type,
                    norm_type,
                )
                for _ in range(q)
            ]
        )

    def forward(self, inpt: Tensor) -> list:
        if inpt.ndim == 3:
            inpt = inpt.unsqueeze(dim=1)
        batch_size, _, seq_len, _ = inpt.shape
        feat_x = self.en(inpt)
        x = feat_x.transpose(-2, -1).contiguous()
        x = x.view(batch_size, -1, seq_len)
        pre_x = inpt.transpose(-2, -1).contiguous()
        out_list = []
        for gag in self.gags:
            pre_x = gag(x, pre_x)
            out_list.append(pre_x)
        return out_list


class GlanceGazeModule(nn.Module):
    def __init__(
        self,
        kd1: int,
        cd1: int,
        d_feat: int,
        p: int,
        dilas: list,
        fft_num: int,
        is_causal: bool,
        is_squeezed: bool,
        acti_type: str,
        norm_type: str,
    ):
        super().__init__()
        self.glance_block = GlanceBlock(
            kd1, cd1, d_feat, p, dilas, fft_num, is_causal, acti_type, norm_type
        )
        self.gaze_block = GazeBlock(
            kd1, cd1, d_feat, p, dilas, fft_num, is_causal, is_squeezed, norm_type
        )

    def forward(self, feat_x: Tensor, pre_x: Tensor) -> Tensor:
        gain_filter = self.glance_block(feat_x, pre_x)
        com_resi = self.gaze_block(feat_x, pre_x)
        pre_mag = torch.norm(pre_x, dim=1)
        pre_phase = torch.atan2(pre_x[:, -1, ...], pre_x[:, 0, ...])
        filtered_x = pre_mag * gain_filter
        coarse_x = torch.stack(
            (filtered_x * torch.cos(pre_phase), filtered_x * torch.sin(pre_phase)),
            dim=1,
        )
        return coarse_x + com_resi


class GlanceBlock(nn.Module):
    def __init__(
        self,
        kd1: int,
        cd1: int,
        d_feat: int,
        p: int,
        dilas: list,
        fft_num: int,
        is_causal: bool,
        acti_type: str,
        norm_type: str,
    ):
        super().__init__()
        ci = (fft_num // 2 + 1) * 2 + d_feat
        self.in_conv_main = nn.Conv1d(ci, d_feat, 1)
        self.in_conv_gate = nn.Sequential(nn.Conv1d(ci, d_feat, 1), nn.Sigmoid())
        self.tcn_g = nn.Sequential(
            *[
                SqueezedTCNGroup(kd1, cd1, d_feat, dilas, is_causal, norm_type)
                for _ in range(p)
            ]
        )
        if acti_type == "sigmoid":
            acti = nn.Sigmoid()
        elif acti_type == "tanh":
            acti = nn.Tanh()
        elif acti_type == "relu":
            acti = nn.ReLU()
        else:
            raise RuntimeError("a activation function must be assigned")
        self.linear_g = nn.Sequential(nn.Conv1d(d_feat, fft_num // 2 + 1, 1), acti)

    def forward(self, feat_x: Tensor, pre_x: Tensor) -> Tensor:
        batch_size, _, _, seq_len = pre_x.shape
        pre_x = pre_x.view(batch_size, -1, seq_len)
        inpt = torch.cat((feat_x, pre_x), dim=1)
        x = self.in_conv_main(inpt) * self.in_conv_gate(inpt)
        x = self.tcn_g(x)
        return self.linear_g(x)


class GazeBlock(nn.Module):
    def __init__(
        self,
        kd1: int,
        cd1: int,
        d_feat: int,
        p: int,
        dilas: list,
        fft_num: int,
        is_causal: bool,
        is_squeezed: bool,
        norm_type: str,
    ):
        super().__init__()
        self.is_squeezed = is_squeezed
        ci = (fft_num // 2 + 1) * 2 + d_feat
        self.in_conv_main = nn.Conv1d(ci, d_feat, 1)
        self.in_conv_gate = nn.Sequential(nn.Conv1d(ci, d_feat, 1), nn.Sigmoid())

        if not is_squeezed:
            self.tcm_r = nn.Sequential(
                *[
                    SqueezedTCNGroup(kd1, cd1, d_feat, dilas, is_causal, norm_type)
                    for _ in range(p)
                ]
            )
            self.tcm_i = nn.Sequential(
                *[
                    SqueezedTCNGroup(kd1, cd1, d_feat, dilas, is_causal, norm_type)
                    for _ in range(p)
                ]
            )
        else:
            self.tcm_ri = nn.Sequential(
                *[
                    SqueezedTCNGroup(kd1, cd1, d_feat, dilas, is_causal, norm_type)
                    for _ in range(p)
                ]
            )

        self.linear_r = nn.Conv1d(d_feat, fft_num // 2 + 1, 1)
        self.linear_i = nn.Conv1d(d_feat, fft_num // 2 + 1, 1)

    def forward(self, feat_x: Tensor, pre_x: Tensor) -> Tensor:
        batch_size, _, _, seq_len = pre_x.shape
        pre_x = pre_x.view(batch_size, -1, seq_len)
        inpt = torch.cat((feat_x, pre_x), dim=1)
        x = self.in_conv_main(inpt) * self.in_conv_gate(inpt)
        if not self.is_squeezed:
            x_r, x_i = self.tcm_r(x), self.tcm_i(x)
        else:
            x = self.tcm_ri(x)
            x_r, x_i = x, x
        x_r, x_i = self.linear_r(x_r), self.linear_i(x_i)
        return torch.stack((x_r, x_i), dim=1)


class SqueezedTCNGroup(nn.Module):
    def __init__(
        self,
        kd1: int,
        cd1: int,
        d_feat: int,
        dilas: list,
        is_causal: bool,
        norm_type: str,
    ):
        super().__init__()
        self.tcns = nn.Sequential(
            *[
                SqueezedTCM(kd1, cd1, d_feat, dilation, is_causal, norm_type)
                for dilation in dilas
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.tcns(x)


class SqueezedTCM(nn.Module):
    def __init__(
        self,
        kd1: int,
        cd1: int,
        d_feat: int,
        dilation: int,
        is_causal: bool,
        norm_type: str,
    ):
        super().__init__()
        self.in_conv = nn.Conv1d(d_feat, cd1, 1, bias=False)
        if is_causal:
            padding = ((kd1 - 1) * dilation, 0)
        else:
            pad = (kd1 - 1) * dilation // 2
            padding = (pad, pad)
        self.d_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            nn.ConstantPad1d(padding, value=0.0),
            nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False),
        )
        self.out_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            nn.Conv1d(cd1, d_feat, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.in_conv(x)
        x = self.d_conv(x)
        x = self.out_conv(x)
        return x + residual


class U2Net_Encoder(nn.Module):
    def __init__(
        self,
        cin: int,
        k1: tuple,
        k2: tuple,
        c: int,
        intra_connect: str,
        norm_type: str,
    ):
        super().__init__()
        k_beg = (2, 5)
        c_end = 64
        self.meta_unet_list = nn.ModuleList(
            [
                En_unet_module(cin, c, k_beg, k2, intra_connect, norm_type, scale=4),
                En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=3),
                En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=2),
                En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=1),
            ]
        )
        self.last_conv = nn.Sequential(
            GateConv2d(c, c_end, k1, (1, 2)),
            NormSwitch(norm_type, "2D", c_end),
            nn.PReLU(c_end),
        )

    def forward(self, x: Tensor) -> Tensor:
        for module in self.meta_unet_list:
            x = module(x)
        return self.last_conv(x)


class UNet_Encoder(nn.Module):
    def __init__(
        self,
        cin: int,
        k1: tuple,
        c: int,
        norm_type: str,
    ):
        super().__init__()
        k_beg = (2, 5)
        c_end = 64
        self.unet_list = nn.ModuleList(
            [
                nn.Sequential(
                    GateConv2d(cin, c, k_beg, (1, 2)),
                    NormSwitch(norm_type, "2D", c),
                    nn.PReLU(c),
                ),
                nn.Sequential(
                    GateConv2d(c, c, k1, (1, 2)),
                    NormSwitch(norm_type, "2D", c),
                    nn.PReLU(c),
                ),
                nn.Sequential(
                    GateConv2d(c, c, k1, (1, 2)),
                    NormSwitch(norm_type, "2D", c),
                    nn.PReLU(c),
                ),
                nn.Sequential(
                    GateConv2d(c, c, k1, (1, 2)),
                    NormSwitch(norm_type, "2D", c),
                    nn.PReLU(c),
                ),
                nn.Sequential(
                    GateConv2d(c, c_end, k1, (1, 2)),
                    NormSwitch(norm_type, "2D", c_end),
                    nn.PReLU(c_end),
                ),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for module in self.unet_list:
            x = module(x)
        return x


class En_unet_module(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        k1: tuple,
        k2: tuple,
        intra_connect: str,
        norm_type: str,
        scale: int,
    ):
        super().__init__()
        self.intra_connect = intra_connect
        self.in_conv = nn.Sequential(
            GateConv2d(cin, cout, k1, (1, 2)),
            NormSwitch(norm_type, "2D", cout),
            nn.PReLU(cout),
        )
        self.enco = nn.ModuleList(
            [Conv2dunit(k2, cout, norm_type) for _ in range(scale)]
        )
        self.deco = nn.ModuleList(
            [
                Deconv2dunit(k2, cout, "add" if index == 0 else intra_connect, norm_type)
                for index in range(scale)
            ]
        )
        self.skip_connect = Skip_connect(intra_connect)

    def forward(self, x: Tensor):
        x_resi = self.in_conv(x)
        x = x_resi
        x_list = []
        for module in self.enco:
            x = module(x)
            x_list.append(x)

        for index, module in enumerate(self.deco):
            if index == 0:
                x = module(x)
            else:
                x = module(self.skip_connect(x, x_list[-(index + 1)]))
        return x_resi + x


class Conv2dunit(nn.Module):
    def __init__(self, kernel_size: tuple, channels: int, norm_type: str):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, (1, 2)),
            NormSwitch(norm_type, "2D", channels),
            nn.PReLU(channels),
        )

    def forward(self, x):
        return self.conv(x)


class Deconv2dunit(nn.Module):
    def __init__(
        self,
        kernel_size: tuple,
        channels: int,
        intra_connect: str,
        norm_type: str,
    ):
        super().__init__()
        if intra_connect == "add":
            deconv = nn.ConvTranspose2d(channels, channels, kernel_size, (1, 2))
        elif intra_connect == "cat":
            deconv = nn.ConvTranspose2d(2 * channels, channels, kernel_size, (1, 2))
        else:
            raise ValueError(f"Unsupported intra_connect: {intra_connect}")
        self.deconv = nn.Sequential(
            deconv,
            NormSwitch(norm_type, "2D", channels),
            nn.PReLU(channels),
        )

    def forward(self, x):
        return self.deconv(x)


class GateConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple,
    ):
        super().__init__()
        kernel_t = kernel_size[0]
        if kernel_t > 1:
            padding = (0, 0, kernel_t - 1, 0)
            self.conv = nn.Sequential(
                nn.ConstantPad2d(padding, value=0.0),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels * 2,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels * 2,
                kernel_size=kernel_size,
                stride=stride,
            )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(dim=1)
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()


class Skip_connect(nn.Module):
    def __init__(self, connect: str):
        super().__init__()
        self.connect = connect

    def forward(self, x_main, x_aux):
        if self.connect == "add":
            return x_main + x_aux
        if self.connect == "cat":
            return torch.cat((x_main, x_aux), dim=1)
        raise ValueError(f"Unsupported connect mode: {self.connect}")


class NormSwitch(nn.Module):
    def __init__(self, norm_type: str, dim_size: str, channels: int):
        super().__init__()
        assert norm_type in ["BN", "IN"] and dim_size in ["1D", "2D"]
        if norm_type == "BN":
            if dim_size == "1D":
                self.norm = nn.BatchNorm1d(channels)
            else:
                self.norm = nn.BatchNorm2d(channels)
        else:
            if dim_size == "1D":
                self.norm = nn.InstanceNorm1d(channels, affine=True)
            else:
                self.norm = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        return self.norm(x)


def stagewise_com_mag_mse_loss(esti_list, label, frame_list):
    alpha_list = [0.1 for _ in range(len(esti_list))]
    alpha_list[-1] = 1
    mask_for_loss = []
    utt_num = label.size()[0]
    with torch.no_grad():
        for index in range(utt_num):
            tmp_mask = torch.ones(
                (frame_list[index], label.size()[-2]), dtype=label.dtype
            )
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(
            label.device
        )
        mask_for_loss = mask_for_loss.transpose(-2, -1).contiguous()
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    loss1 = 0.0
    loss2 = 0.0
    mag_label = torch.norm(label, dim=1)
    for index, estimate in enumerate(esti_list):
        mag_esti = torch.norm(estimate, dim=1)
        loss1 = loss1 + alpha_list[index] * (
            ((estimate - label) ** 2.0) * com_mask_for_loss
        ).sum() / com_mask_for_loss.sum()
        loss2 = loss2 + alpha_list[index] * (
            ((mag_esti - mag_label) ** 2.0) * mask_for_loss
        ).sum() / mask_for_loss.sum()
    return 0.5 * (loss1 + loss2)


class GaGNetModule(pl.LightningModule):
    def __init__(
        self,
        lr=1e-3,
        weight_decay=1e-5,
        metrics=None,
        train_metrics=None,
        val_metrics=None,
        test_metrics=None,
    ):
        super().__init__()
        self.model = GaGNet()
        self.lr = lr
        self.weight_decay = weight_decay
        sample_rate = 16000
        self.fft_num = 320
        self.win_size = int(0.020 * sample_rate)
        self.win_shift = int(0.010 * sample_rate)
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

    def stft(self, wav):
        wav = wav.squeeze(1)
        stft = torch.stft(
            wav,
            self.fft_num,
            self.win_shift,
            self.win_size,
            torch.hann_window(self.win_size).to(wav.device),
            return_complex=True,
        )
        stft = torch.view_as_real(stft)
        stft = stft.permute(0, 3, 2, 1).contiguous()
        mag = torch.norm(stft, dim=1)
        phase = torch.atan2(stft[:, -1, ...], stft[:, 0, ...])
        return torch.stack((mag * torch.cos(phase), mag * torch.sin(phase)), dim=1)

    def istft(self, stft, wav_len):
        stft = torch.view_as_complex(stft.permute(0, 2, 3, 1).contiguous())
        wav = torch.istft(
            stft,
            self.fft_num,
            self.win_shift,
            self.win_size,
            torch.hann_window(self.win_size).to(stft.device),
            length=wav_len,
        )
        return wav.unsqueeze(1)

    def forward(self, acm, bcm):
        noisy_stft = self.stft(acm)
        enhanced_stft = self.model(noisy_stft)
        enhanced_wav = self.istft(enhanced_stft[-1], acm.size(-1))
        return enhanced_wav, enhanced_stft

    def training_step(self, batch, batch_idx):
        target, acm, bcm = batch
        enhanced_wav, enhanced_stft = self(acm, bcm)
        target_stft = self.stft(target).permute(0, 1, 3, 2).contiguous()
        frame_list = [acm.size(-1) // self.win_shift + 1 for _ in range(acm.size(0))]
        loss = stagewise_com_mag_mse_loss(enhanced_stft, target_stft, frame_list)
        self.log("train_loss", loss)
        return {
            "loss": loss,
            "output": enhanced_wav,
            "target": target,
            "acm": acm,
            "bcm": bcm,
        }

    def validation_step(self, batch, batch_idx):
        target, acm, bcm = batch
        enhanced_wav, enhanced_stft = self(acm, bcm)
        target_stft = self.stft(target).permute(0, 1, 3, 2).contiguous()
        frame_list = [acm.size(-1) // self.win_shift + 1 for _ in range(acm.size(0))]
        loss = stagewise_com_mag_mse_loss(enhanced_stft, target_stft, frame_list)
        self.log("val_loss", loss)
        return {
            "loss": loss,
            "output": enhanced_wav,
            "target": target,
            "acm": acm,
            "bcm": bcm,
        }

    def test_step(self, batch, batch_idx):
        target, acm, bcm = batch
        enhanced_wav, enhanced_stft = self(acm, bcm)
        target_stft = self.stft(target).permute(0, 1, 3, 2).contiguous()
        frame_list = [acm.size(-1) // self.win_shift + 1 for _ in range(acm.size(0))]
        loss = stagewise_com_mag_mse_loss(enhanced_stft, target_stft, frame_list)
        self.log("test_loss", loss)
        return {
            "loss": loss,
            "output": enhanced_wav,
            "target": target,
            "acm": acm,
            "bcm": bcm,
        }

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

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
