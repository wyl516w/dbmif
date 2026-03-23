#!/usr/bin/env python3
# dual_enhance_wave_fixed.py
import math, torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection


# ---------------- 1. 相对位置编码 ----------------
class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        i = torch.arange(0, d_model, 2)
        angle = pos / torch.pow(10000.0, i / d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(angle), torch.cos(angle)
        self.register_buffer("pe", pe)

    def forward(self, x):  # x: [B,T,D]
        return self.dropout(x + self.pe[: x.size(1)])


# ---------------- 2. Feed-Forward ----------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        inner = d_model * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, inner),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inner, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + 0.5 * self.net(x)  # Macaron


# ---------------- 3. Depthwise Conv ----------------
class DepthwiseConvModule(nn.Module):
    def __init__(self, d_model: int, k: int = 15, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.dw = nn.Conv1d(d_model, d_model, k, padding=k // 2, groups=d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, 1)
        self.act = nn.SiLU()
        self.do = nn.Dropout(dropout)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):  # x: [B,T,D]
        y = self.ln(x).transpose(1, 2)  # [B,D,T]
        y = self.glu(self.pw1(y))
        y = self.dw(y)
        y = self.act(self.pw2(y)).transpose(1, 2)
        return x + self.do(y)


# ---------------- 4. Self-Attention ----------------
class MultiHeadSelfAttentionRel(nn.Module):
    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.do = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        y, _ = self.attn(*(self.ln(x),) * 3, key_padding_mask=mask)
        return x + self.do(y)


# ---------------- 5. Conformer Block ----------------
class ConformerBlock(nn.Module):
    def __init__(self, d_model: int = 96, heads: int = 4, exp: int = 2, k: int = 15, drop: float = 0.1):
        super().__init__()
        self.ff1 = FeedForward(d_model, exp, drop)
        self.mha = MultiHeadSelfAttentionRel(d_model, heads, drop)
        self.conv = DepthwiseConvModule(d_model, k, drop)
        self.ff2 = FeedForward(d_model, exp, drop)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.ff1(x)
        x = self.mha(x, mask)
        x = self.conv(x)
        x = self.ff2(x)
        return self.ln(x)


# ---------------- 6. Conv2d 下采样 ----------------
class Conv2dSub(nn.Module):
    def __init__(self, in_ch: int, d_model: int, freq_bins: int, drop: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, d_model, 3, stride=2, padding=(0, 1)),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, 3, stride=2, padding=(0, 1)),
            nn.SiLU(),
        )
        self.out_F = math.ceil(freq_bins / 4)
        self.proj = nn.Linear(d_model * self.out_F, d_model, bias=False)
        self.pos = RelPositionalEncoding(d_model, drop)

    def forward(self, x):  # x: [B*C,1,T,F]
        x = self.conv(x)  # [B*C,d,T/4,F/4]
        B, d, T4, F4 = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T4, d * F4)
        return self.pos(self.proj(x))  # [B*C,T/4,D]


# ---------------- 7. Encoder ----------------
class Encoder(nn.Module):
    def __init__(self, layers: int, d_model: int, freq_bins: int):
        super().__init__()
        self.sub = Conv2dSub(1, d_model, freq_bins)
        self.stk = nn.ModuleList([ConformerBlock(d_model=d_model) for _ in range(layers)])

    def forward(self, x):  # x: [B*C,1,T,F]
        x = self.sub(x)
        for blk in self.stk:
            x = blk(x)
        return x  # [B*C,T/4,D]


# ---------------- 8. Gate Fusion ----------------
class GateFusion(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(d_model))

    def forward(self, h):  # h: [B,2,T,D]
        h_air, h_bone = h[:, 0], h[:, 1]
        logits = torch.einsum("btd,d->bt", h_air - h_bone, self.v) / math.sqrt(self.v.numel())
        g = torch.sigmoid(logits).unsqueeze(-1)
        return g * h_air + (1 - g) * h_bone, g.mean(dim=1, keepdim=True)


# ---------------- 9. Decoder ----------------
class Decoder(nn.Module):
    def __init__(self, layers: int, d_model: int, F: int):
        super().__init__()
        self.F, self.K = F, 8
        self.stk = nn.ModuleList([ConformerBlock(d_model=d_model) for _ in range(layers)])
        self.up1 = nn.ConvTranspose1d(d_model, d_model, 4, 2, 1)
        self.up2 = nn.ConvTranspose1d(d_model, d_model, 4, 2, 1)
        self.head = nn.Conv1d(d_model, F * self.K, 1)

    def forward(self, x):  # x: [B,T/4,D]
        for blk in self.stk:
            x = blk(x)
        x = F.silu(self.up1(x.transpose(1, 2)))
        x = F.silu(self.up2(x))
        x = self.head(x)  # [B,F*K,T]
        B, _, T = x.shape
        return x.view(B, self.F, self.K, T).permute(0, 1, 3, 2)  # [B,F,T,8]


# ----------------10. Channel Attention --------
class ChannelAttention(nn.Module):
    def __init__(self, C: int = 4, r: int = 4):
        super().__init__()
        mid = max(1, C // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(C, mid, bias=False), nn.ReLU(inplace=True), nn.Linear(mid, C, bias=False), nn.Softmax(dim=1))

    def forward(self, lst):
        x = torch.stack(lst, dim=1)  # [B,4,F,T]
        B, C, F, T = x.shape
        w = self.fc(self.pool(x.permute(0, 1, 3, 2)).view(B, C)).view(B, C, 1, 1)
        return (x * w).sum(dim=1)  # [B,F,T]


# ----------------11. DualEnhancerWave ----------
class DualEnhancerWave(nn.Module):
    def __init__(self, n_fft=512, hop=128, win=512, d_model=96, enc_layers=2, dec_layers=2):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.F = n_fft // 2 + 1
        self.register_buffer("window", torch.hann_window(win))

        self.enc_air = Encoder(enc_layers, d_model, self.F)
        self.enc_bone = Encoder(enc_layers, d_model, self.F)
        self.gate = GateFusion(d_model)
        self.dec = Decoder(dec_layers, d_model, self.F)
        self.ca = ChannelAttention(4)

    # --- STFT helpers ---
    def _wav2spec(self, w):  # w: [B,C,T]
        B, C, T = w.shape
        S = torch.stft(w.view(B * C, T), self.n_fft, self.hop, self.win, window=self.window, return_complex=True)
        return S.view(B, C, self.F, -1)  # [B,C,F,T]

    def _spec2wav(self, S, B):
        return torch.istft(S.view(B, self.F, -1), self.n_fft, self.hop, self.win, window=self.window).unsqueeze(1)

    # --- forward ---
    def forward(self, wav_air, wav_bone):
        B = wav_air.size(0)
        spec_air = self._wav2spec(wav_air)[:, 0]  # [B,F,T]
        spec_bone = self._wav2spec(wav_bone)[:, 0]

        # permute (F,T)->(T,F) then encode
        h_air = self.enc_air(spec_air.real.permute(0, 2, 1).unsqueeze(1))
        h_bone = self.enc_bone(spec_bone.real.permute(0, 2, 1).unsqueeze(1))
        fused, _ = self.gate(torch.stack([h_air, h_bone], 1))

        out = self.dec(fused)  # [B,F,T',8]

        # ----- 对齐时间维 -----
        T_ref = spec_air.size(-1)
        if out.size(2) > T_ref:
            out = out[:, :, :T_ref, :]
        elif out.size(2) < T_ref:
            out = F.pad(out, (0, 0, 0, T_ref - out.size(2)))

        air_r, air_i, bone_r, bone_i, sh_r, sh_i, dir_r, dir_i = out.unbind(-1)
        mag_air, mag_bone = spec_air.abs(), spec_bone.abs()
        real_ref, imag_ref = spec_air.real, spec_air.imag
        real_bone_ref, imag_bone_ref = spec_bone.real, spec_bone.imag

        real_air = real_ref + torch.tanh(air_r) * mag_air
        imag_air = imag_ref + torch.tanh(air_i) * mag_air
        real_bone = real_bone_ref + torch.tanh(bone_r) * mag_bone
        imag_bone = imag_bone_ref + torch.tanh(bone_i) * mag_bone
        mag_sh = 0.5 * (mag_air + mag_bone)
        real_sh = real_ref + torch.tanh(sh_r) * mag_sh
        imag_sh = imag_ref + torch.tanh(sh_i) * mag_sh
        real_dir, imag_dir = dir_r, dir_i

        real_fused = self.ca([real_air, real_bone, real_sh, real_dir])
        imag_fused = self.ca([imag_air, imag_bone, imag_sh, imag_dir])
        spec_fused = torch.complex(real_fused, imag_fused)

        enh = self._spec2wav(spec_fused, B)  # [B,1,T]
        return enh, spec_fused


class ConformerBasedEnhancer(pl.LightningModule):

    def __init__(
        self,
        n_fft: int = 512,
        hop: int = 128,
        win: int = 512,
        model_dim: int = 128,
        num_blocks: int = 2,
        lr: float = 1e-3,
        metrics: MetricCollection | None = None,
        train_metrics: MetricCollection | None = None,
        val_metrics: MetricCollection | None = None,
        test_metrics: MetricCollection | None = None,
    ):
        """
        Lightning 模块封装：
          - 包含模型、STFT、损失计算以及训练/验证/测试步骤
        """
        super().__init__()
        self.lr = lr

        self.model = DualEnhancerWave(n_fft, hop, win, model_dim, num_blocks, num_blocks)

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

    def forward(self, waveform1, waveform2):
        return self.model(waveform1, waveform2)

    def compute_loss(self, enh, clean):
        """
        损失函数：采用实部和虚部的 L1 损失 + 幅度损失
        """
        mag_enh, mag_cln = enh.abs(), clean.abs()
        return F.l1_loss(enh, clean) + 0.5 * F.mse_loss(mag_enh, mag_cln)

    def training_step(self, batch, batch_idx):
        reference, corrupted_ac, corrupted_bc = batch
        enhanced, signal = self(corrupted_ac, corrupted_bc)
        ref = self.model._wav2spec(reference).squeeze(1)
        loss = self.compute_loss(signal, ref)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return {"loss": loss, "enhanced": enhanced, "reference": reference, "corrupted_ac": corrupted_ac, "corrupted_bc": corrupted_bc}

    def validation_step(self, batch, batch_idx):
        reference, corrupted_ac, corrupted_bc = batch
        enhanced, signal = self(corrupted_ac, corrupted_bc)
        ref = self.model._wav2spec(reference).squeeze(1)
        loss = self.compute_loss(signal, ref)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return {"loss": loss, "enhanced": enhanced, "reference": reference, "corrupted_ac": corrupted_ac, "corrupted_bc": corrupted_bc}

    def test_step(self, batch, batch_idx):
        reference, corrupted_ac, corrupted_bc = batch
        enhanced, signal = self(corrupted_ac, corrupted_bc)
        ref = self.model._wav2spec(reference).squeeze(1)
        loss = self.compute_loss(signal, ref)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        return {"loss": loss, "enhanced": enhanced, "reference": reference, "corrupted_ac": corrupted_ac, "corrupted_bc": corrupted_bc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

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

if __name__ == "__main__":
    B, C, T = 2, 1, 16000
    ref = torch.randn(B, C, T)
    
    ac = torch.randn(B, C, T)
    bc = torch.randn(B, C, T)
    model = ConformerBasedEnhancer()
    loss = model.training_step((ref, ac, bc), 0)
    print(loss["loss"])  # (B,1,T)
    print(loss["enhanced"].shape)  # (B,1,T)
    print(loss["reference"].shape)  # (B,1,T)
    print(loss["corrupted_ac"].shape)  # (B,1,T)
    print(loss["corrupted_bc"].shape)  # (B,1,T)
