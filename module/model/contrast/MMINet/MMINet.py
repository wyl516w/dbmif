import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math


# ---------------------------
# SI-SNR 损失（取负值作为 loss）
def SI_SNR(_s, s, zero_mean=True, eps=1e-8):
    """
    计算两个音频之间的 SI-SNR 指标
    输入：
         _s: 模型输出，形状为 [B, 1, N]
         s: 参考信号，形状为 [B, 1, N]
    """
    # 将 [B, 1, N] 转换为 [B, N]
    if _s.dim() == 3 and _s.size(1) == 1:
        _s = _s.squeeze(1)
    if s.dim() == 3 and s.size(1) == 1:
        s = s.squeeze(1)

    if zero_mean:
        _s = _s - torch.mean(_s, dim=1, keepdim=True)
        s = s - torch.mean(s, dim=1, keepdim=True)

    s_target = (
        torch.sum(_s * s, dim=1, keepdim=True)
        * s
        / (torch.norm(s, p=2, dim=1, keepdim=True) ** 2 + eps)
    )
    e_noise = _s - s_target
    si_snr_val = 20 * torch.log10(
        torch.norm(s_target, p=2, dim=1) / (torch.norm(e_noise, p=2, dim=1) + eps)
    )
    return si_snr_val


def sisnr(x, s, eps=1e-8):
    """
    计算 SI-SNR 值
    x: 分离信号，输入形状为 [B, 1, N]，转换为 [B, N]
    s: 参考信号，输入形状为 [B, 1, N]，转换为 [B, N]
    """
    if x.dim() == 3 and x.size(1) == 1:
        x = x.squeeze(1)
    if s.dim() == 3 and s.size(1) == 1:
        s = s.squeeze(1)

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimension mismatch when calculating si-snr, {} vs {}".format(x.shape, s.shape)
        )

    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = (
        torch.sum(x_zm * s_zm, dim=-1, keepdim=True)
        * s_zm
        / (l2norm(s_zm, keepdim=True) ** 2 + eps)
    )
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))


def si_snr_loss(est, ref, eps=1e-8):
    """
    计算 SI-SNR 损失
    输入：
         est: 模型输出，形状为 [B, 1, N]
         ref: 参考信号，形状为 [B, 1, N]
    先将两者 squeeze 成 [B, N]，再计算每个样本的 SI-SNR 值，最后取负均值作为损失
    """
    if est.dim() == 3 and est.size(1) == 1:
        est = est.squeeze(1)
    if ref.dim() == 3 and ref.size(1) == 1:
        ref = ref.squeeze(1)

    si_snr_val = sisnr(est, ref, eps=eps)
    loss = -torch.mean(si_snr_val)
    return loss


def l2_loss(est, ref):
    """
    计算 L1 损失
    输入：
         est: 模型输出，形状为 [B, 1, N]
         ref: 参考信号，形状为 [B, 1, N]
    先将两者 squeeze 成 [B, N]，再计算 L1 损失
    """
    if est.dim() == 3 and est.size(1) == 1:
        est = est.squeeze(1)
    if ref.dim() == 3 and ref.size(1) == 1:
        ref = ref.squeeze(1)

    loss = nn.MSELoss()(est, ref)
    return loss


# ---------------------------
# 1. Involution1D（支持 stride、dilation、bias 等参数）
# ---------------------------

from .involution import Involution1D


# ---------------------------
# 2. InvolutionBlock（返回 main_out 与 skip 两分支）
# ---------------------------
class InvolutionBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, dilation=1, bias=False):
        """
        Args:
            channels: 输入通道数
            kernel_size: Involution 核大小
            dilation: 扩张系数
            reduction_ratio: 动态核生成时的通道压缩比例
            bias: 是否使用偏置
        """
        super(InvolutionBlock, self).__init__()
        self.involution = Involution1D(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
            padding=kernel_size // 2 * dilation,
        )
        self.prelu = nn.PReLU()
        # LayerNorm 要求输入形状为 (B, L, C)，因此后续需转置
        self.ln = nn.LayerNorm(channels, elementwise_affine=True)
        self.conv_main = nn.Conv1d(channels, channels, kernel_size=1, bias=bias, dilation=dilation)
        self.conv_skip = nn.Conv1d(channels, channels, kernel_size=1, bias=bias, dilation=dilation)

    def forward(self, x):
        """
        Args:
            x: (B, C, L)
        Returns:
            main_out: (B, C, L) = conv_main(out) * x
            skip: (B, C, L) = conv_skip(out)
        """
        out = self.involution(x)  # (B, C, L)
        out = self.prelu(out)
        out = out.transpose(1, 2)  # 转换为 (B, L, C)
        out = self.ln(out)
        out = out.transpose(1, 2)  # 恢复为 (B, C, L)
        main_out = self.conv_main(out) + x
        skip = self.conv_skip(out)
        return main_out, skip


# ---------------------------
# 3. MaskEstimator（前置 LN+Conv，层间 main_out 传递，skip 累加，后处理 PreLU+Conv+ReLU）
# ---------------------------
class MaskEstimator(nn.Module):
    def __init__(
        self,
        channels,
        num_blocks=8,
        num_repeats=3,
        kernel_size=7,
        base_dilation=2,
        bias=False,
    ):
        """
        Args:
            channels: 与 Encoder 输出一致的通道数
            num_blocks: 每组 InvolutionBlock 数量
            num_repeats: 重复组数（迭代次数）
            kernel_size: Involution 核大小
            base_dilation: 组内基础扩张系数，dilation = base_dilation ** (i % num_blocks)
            reduction_ratio: 动态核生成时的通道压缩比例
            bias: 是否使用偏置
        """
        super(MaskEstimator, self).__init__()
        # 前置处理：LayerNorm（需转置）+ 1×1 Conv
        self.initial_ln = nn.LayerNorm(channels)
        self.initial_conv = nn.Conv1d(channels, channels, kernel_size=1, bias=bias)

        self.blocks = nn.ModuleList()
        total_layers = num_repeats * num_blocks
        for i in range(total_layers):
            dilation = base_dilation ** (i % num_blocks)
            self.blocks.append(
                InvolutionBlock(
                    channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    bias=bias,
                )
            )

        # 后置处理：PreLU + 1×1 Conv + ReLU（生成非负 ratio mask）
        self.post_prelu = nn.PReLU()
        self.post_conv = nn.Conv1d(channels, channels, kernel_size=1, bias=bias)
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: (B, C, L) Encoder 输出
        Returns:
            mask: (B, C, L) 非负 ratio mask
        """
        # 前置处理：LayerNorm（先转置）+ 1×1 Conv
        x_ln = self.initial_ln(x.transpose(1, 2)).transpose(1, 2)
        x_proc = self.initial_conv(x_ln)

        skip_accum = 0
        out = x_proc
        for block in self.blocks:
            main_out, skip = block(out)
            skip_accum = skip_accum + skip  # 累加所有 skip 输出
            out = main_out  # main_out 传递到下一层

        out_post = self.post_prelu(skip_accum)
        out_post = self.post_conv(out_post)
        mask = self.post_relu(out_post)
        return mask


class Encoder(nn.Module):
    """
    Encoder E that linearly merges two-channel inputs (AC & BC) into a single
    feature map z using a 1-D convolution with N kernels of size (2, L).

    Arguments:
    - in_channels: number of input channels (2 for AC+BC)
    - out_channels: N (number of learnable filters)
    - kernel_size: L (length of each filter in samples)
    - stride: frame shift (how many samples we move between frames)
    - padding: optional zero-padding to ensure consistent framing
    """

    def __init__(self, in_channels=2, out_channels=256, kernel_size=16, stride=8, padding=0):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # U in R^{out_channels x in_channels x kernel_size}
        # This is the learnable weight matrix that performs the linear transform
        self.U = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size) * 0.01)

    def forward(self, x):
        """
        x shape: [B, in_channels, T]  (e.g., B=batch size, 2=AC+BC, T=time)
        Returns z shape: [B, out_channels, K], where K is the number of frames.
        """

        B, C, T = x.shape
        # Check we have the expected number of channels
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"
        x_padded = F.pad(x, (self.padding, self.padding))  # (left, right) in 1D
        x_frames = self.unfold_1d(x_padded, kernel_size=self.kernel_size, stride=self.stride)
        B_, C_, K, L_ = x_frames.shape  # B_, C_, K, L_ = B, in_channels, K, kernel_size
        x_frames_reshaped = x_frames.transpose(2, 3).reshape(B_, C_ * L_, K)  # => [B, (C*L), K]
        U_reshaped = self.U.view(self.out_channels, -1)  # => [out_channels, (C*L)]
        z = torch.einsum("ol, blk -> bok", U_reshaped, x_frames_reshaped)
        return z

    def unfold_1d(self, x, kernel_size, stride=1):
        """
        Unfold 1D input x into overlapping frames of size kernel_size with given stride.
        x shape: [B, C, T]
        Returns: [B, C, K, kernel_size], where K is the number of frames
        """
        B, C, T = x.shape

        # Number of frames
        # e.g. if T=16, kernel_size=16, stride=8 => K= (16 - 16)//8 + 1 = 1
        # for a general formula:
        K = (T - kernel_size) // stride + 1
        frames = []
        for i in range(K):
            start = i * stride
            end = start + kernel_size
            frame_i = x[:, :, start:end]  # [B, C, kernel_size]
            frames.append(frame_i)
        # Stack along new dimension K => [B, C, K, kernel_size]
        x_frames = torch.stack(frames, dim=2)
        return x_frames


class Decoder(nn.Module):
    """
    A simple linear decoder D:
      Y = V c,
    followed by overlap-add to produce a single-channel waveform.

    Args:
      in_features (int): N, the feature dimension from the encoder.
      kernel_size (int): L, the frame length to reconstruct.
      stride (int): the hop size (frame shift).
    """

    def __init__(self, in_features=256, kernel_size=16, stride=8):  # N  # L
        super(Decoder, self).__init__()
        self.in_features = in_features
        self.kernel_size = kernel_size
        self.stride = stride

        # V in R^{L x N}
        # Each row of V is an L-dimensional time-domain point,
        # each column corresponds to one feature dimension in c.
        self.V = nn.Parameter(torch.randn(kernel_size, in_features) * 0.01)

    def forward(self, c):
        """
        c shape: [B, N, K]
          B = batch size
          N = feature dimension
          K = number of frames

        Returns a single-channel waveform:
          y shape: [B, 1, T_out]
          where T_out = (K-1)*stride + kernel_size
        """
        B, N, K = c.shape
        L = self.kernel_size
        hop = self.stride

        # Prepare output buffer for overlap-add
        T_out = (K - 1) * hop + L
        y = torch.zeros(B, 1, T_out, device=c.device, dtype=c.dtype)

        # For each frame k:
        for k_idx in range(K):
            # c[:, :, k_idx] => [B, N]
            # Linear transform => [B, L]
            #   Y_k = V * c_k
            #   V: [L, N], c_k: [B, N] => multiply over N => [B, L]
            frame_out = torch.einsum("ln,bn->bl", self.V, c[:, :, k_idx])
            # Overlap-add into y
            start = k_idx * hop
            end = start + L
            # y[:, 0, start:end] => [B, L]
            y[:, 0, start:end] += frame_out

        return y


# ---------------------------
# 6. MMINet（端到端多模态语音增强模型）
# ---------------------------


class MMINet(pl.LightningModule):
    def __init__(
        self,
        encoder_channels=256,
        num_blocks=8,
        num_repeats=3,
        invo_kernel_size=3,
        base_dilation=2,
        bias=False,
        learning_rate=2e-4,
        weight_decay=1e-4,
        peak_lr=2e-4,
        final_lr=2e-5,
        metrics=None,
        train_metrics=None,
        val_metrics=None,
        test_metrics=None,
    ):
        """
        Args:
            encoder_channels: Encoder 输出通道数
            num_blocks: MaskEstimator 中每组 InvolutionBlock 数量
            num_repeats: MaskEstimator 迭代组数
            invo_kernel_size: Involution 核大小
            base_dilation: MaskEstimator 中基础扩张系数
            bias: 是否使用偏置
            learning_rate: 学习率（备用）
            weight_decay: 权重衰减
            peak_lr: 初始学习率
            final_lr: 最终学习率
            metrics, train_metrics, val_metrics, test_metrics: 用于评估日志的指标对象，支持 clone(prefix) 方法
        """
        super(MMINet, self).__init__()
        self.encoder = Encoder(
            in_channels=2, out_channels=encoder_channels, kernel_size=16, stride=8
        )
        self.mask_estimator = MaskEstimator(
            encoder_channels,
            num_blocks,
            num_repeats,
            kernel_size=invo_kernel_size,
            base_dilation=base_dilation,
            bias=bias,
        )
        self.decoder = Decoder(
            in_features=encoder_channels,
            kernel_size=16,
            stride=16 // 2,
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.peak_lr = peak_lr
        self.final_lr = final_lr

        if metrics is not None:
            self.train_metrics = metrics.clone(prefix="train/")
            self.val_metrics = metrics.clone(prefix="val/")
            self.test_metrics = metrics.clone(prefix="test/")
        if train_metrics is not None:
            self.train_metrics = train_metrics.clone(prefix="train/")
        if val_metrics is not None:
            self.val_metrics = val_metrics.clone(prefix="val/")
        if test_metrics is not None:
            self.test_metrics = test_metrics.clone(prefix="test/")

    def forward(self, ac, bc):
        """
        Args:
            ac: 噪声干扰的 AC 语音 (B, 1, T)
            bc: 同步的 BC 语音 (B, 1, T)
        Returns:
            enhanced: 增强后的时域语音 (B, 1, T_out)
        """
        x = torch.cat([ac, bc], dim=1)  # 拼接成 (B, 2, T)
        z = self.encoder(x)  # 得到融合特征 (B, encoder_channels, L)
        mask = self.mask_estimator(z)  # 得到 ratio mask (B, encoder_channels, L)
        c = z * mask  # 特征滤波
        enhanced = self.decoder(c)  # 重构时域语音
        return enhanced

    def training_step(self, batch, batch_idx):
        # 假设 batch 为 (ac, bc, clean)，形状均为 (B, 1, T)
        clean, ac, bc = batch
        enhanced = self.forward(ac, bc)
        loss = l2_loss(enhanced, clean)
        # 返回包含额外信息的字典，便于日志记录
        output = {
            "loss": loss,
            "enhanced": enhanced,
            "reference": clean,
            "corrupted_ac": ac,
            "corrupted_bc": bc,
        }
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        clean, ac, bc = batch
        enhanced = self.forward(ac, bc)
        loss = l2_loss(enhanced, clean)
        output = {
            "loss": loss,
            "enhanced": enhanced,
            "reference": clean,
            "corrupted_ac": ac,
            "corrupted_bc": bc,
        }
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return output

    def test_step(self, batch, batch_idx):
        ac, bc, clean = batch
        enhanced = self.forward(ac, bc)
        loss = l2_loss(enhanced, clean)
        output = {
            "loss": loss,
            "enhanced": enhanced,
            "reference": clean,
            "corrupted_ac": ac,
            "corrupted_bc": bc,
        }
        self.log("test_loss", loss)
        return output

    def logging(
        self,
        outputs,
        metrics_collection,
        prefix="all",
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    ):
        # 若 outputs 中包含 loss_dict 则记录
        if outputs.get("loss_dict"):
            for metricname, metricvalue in outputs["loss_dict"].items():
                self.log(
                    prefix + "/" + metricname,
                    metricvalue,
                    on_epoch=False,
                    on_step=True,
                    sync_dist=sync_dist,
                )
        origin_ac_metrics = metrics_collection(outputs["corrupted_ac"], outputs["reference"])
        self.log_dict(
            {k + "/origin_ac": v for k, v in origin_ac_metrics.items()},
            on_epoch=on_epoch,
            on_step=on_step,
            sync_dist=sync_dist,
        )
        origin_bc_metrics = metrics_collection(outputs["corrupted_bc"], outputs["reference"])
        self.log_dict(
            {k + "/origin_bc": v for k, v in origin_bc_metrics.items()},
            on_epoch=on_epoch,
            on_step=on_step,
            sync_dist=sync_dist,
        )
        enhanced_metrics = metrics_collection(outputs["enhanced"], outputs["reference"])
        self.log_dict(
            {k + "/enhanced": v for k, v in enhanced_metrics.items()},
            on_epoch=on_epoch,
            on_step=on_step,
            sync_dist=sync_dist,
        )

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if hasattr(self, "train_metrics"):
            self.logging(
                outputs,
                self.train_metrics,
                prefix="train",
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

    def on_validation_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if hasattr(self, "val_metrics"):
            self.logging(
                outputs,
                self.val_metrics,
                prefix="val",
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

    def on_test_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if hasattr(self, "test_metrics"):
            self.logging(
                outputs,
                self.test_metrics,
                prefix="test",
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # 假设我们监控的是验证损失，验证损失越小越好
            factor=0.5,  # 学习率减半
            patience=4,  # 连续4个epoch无改善时触发
        )
        # Lightning 需要指定监控的指标名称，例如 "val_loss"
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


# ---------------------------
# 测试代码（示例）
# ---------------------------
if __name__ == "__main__":
    batch_size = 4
    T = 16000  # 例如 1 秒语音采样点数
    ac = torch.zeros(batch_size, 1, T)
    ac[:, :, 1] = 1
    bc = torch.zeros(batch_size, 1, T)
    bc[:, :, 1] = 1
    clean = torch.zeros(batch_size, 1, T)
    clean[:, :, 1] = 1

    # 构造 Lightning 模型，参数单独传入，同时传入 metrics（示例中为 None，可根据实际替换）
    model = MMINet(
        encoder_channels=256,
        num_blocks=8,
        num_repeats=3,
        invo_kernel_size=3,
        base_dilation=2,
        bias=False,
        learning_rate=2e-4,
        weight_decay=1e-4,
        peak_lr=2e-4,
        final_lr=2e-5,
        metrics=None,
        train_metrics=None,
        val_metrics=None,
        test_metrics=None,
    )

    # 前向测试
    enhanced = model(ac, bc)
    print("Enhanced shape:", enhanced.shape)
