import torch
from torch import nn
from .jacobian import jac_loss_estimate
from .utils import list2vec, vec2list
from .solver import SOLVERDICT
from .block_fusion import FUSIONBLOCKDICT
from .block_modality import MODALITYBLOCKDICT
from .utils import list2vec_BT, vec2list_BT


class DEQEQFusionModule(nn.Module):

    def __init__(
        self,
        num_out_dims,
        kernel_size=3,
        bias=False,
        fblock_type="smallgatedblock",
        bblock_type="deqfusionblock",
    ):
        super(DEQEQFusionModule, self).__init__()
        self.num_branches = len(num_out_dims)
        self.block = nn.ModuleList([MODALITYBLOCKDICT[fblock_type](num_out_dims[i], kernel_size, bias) for i in range(self.num_branches - 1)])
        self.fusion_block = FUSIONBLOCKDICT[bblock_type](num_out_dims, kernel_size, bias)

    def forward(self, x, injection):
        x_block_out = []
        for i in range(self.num_branches - 1):
            out = self.block[i](x[i], injection[i])
            x_block_out.append(out)
        x_block_out.append(self.fusion_block(x[-1], x_block_out, injection[-1]))
        return x_block_out


class DEQEQFusionBlock(nn.Module):
    # DEQEQFusionBlock
    def __init__(
        self,
        fusion_channels=[50, 50],
        kernel_size=3,
        bias=False,
        stop_mod="abs",  # "rel"
        stop_val=1e-5,
        fsolver="anderson",
        fthresh=105,  # 35
        fblock_type="smallgatedblock",
        bsolver="anderson",
        bthresh=106,  # 36
        bblock_type="deqfusionblock",
    ):
        """
        DEQEQFusionBlock is a fusion block that uses DEQ to fuse features from different modalities.
        Arguments:
            fusion_channels: List of channels for each modality, last one is the fusion channel.
            kernel_size: Kernel size for the convolutional layers.
            bias: Whether to use bias in the convolutional layers.
            stop_mod: Stopping mode for DEQ solver.
            stop_val: Stopping value for DEQ solver.
            fsolver: Solver type for modality solver.
            fthresh: Threshold for modality solver.
            fblock_type: Type of block used in the modality solver.
            bsolver: Solver type for fusion solver.
            bthresh: Threshold for fusion solver.
            bblock_type: Type of block used in the fusion solver.
        """
        super(DEQEQFusionBlock, self).__init__()
        self.func_ = DEQEQFusionModule(
            num_out_dims=fusion_channels,
            kernel_size=kernel_size,
            bias=bias,
            fblock_type=fblock_type.lower() if isinstance(fblock_type, str) else fblock_type,
            bblock_type=bblock_type.lower() if isinstance(bblock_type, str) else bblock_type,
        )
        self.f_thres = fthresh
        self.b_thres = bthresh
        self.stop_mode = stop_mod
        self.stop_val = stop_val
        self.f_solver = SOLVERDICT.get(fsolver.lower()) if isinstance(fsolver, str) else fsolver
        self.b_solver = SOLVERDICT.get(bsolver.lower()) if isinstance(bsolver, str) else bsolver
        if self.f_solver is None:
            Warning(f"Unknown f_solver: {fsolver}, using default solver.")
            self.f_solver = SOLVERDICT["anderson"]
        if self.b_solver is None:
            Warning(f"Unknown b_solver: {bsolver}, using default solver.")
            self.b_solver = SOLVERDICT["anderson"]
        self.hook = None

    def featureFusion(self, features, compute_jac_loss=True):
        device = features[0].device
        x_list = list(features)
        # z_list = [torch.zeros(feature.shape).to(device) for feature in x_list]
        # cutoffs = [(elem.size(1), elem.size(2)) for elem in z_list]
        
        # z1 = list2vec(z_list)
        # func = lambda z: list2vec(self.func_(vec2list(z, cutoffs), x_list))
        z_list = [torch.zeros_like(feature) for feature in x_list]  # (B, C_i, T)
        z1, bt_shape, cutoffs_c = list2vec_BT(z_list)               # (B*T, sumC, 1)

        def func(z_vec):
            # (B*T, sumC, 1) -> [ (B, C_i, T) ... ]
            z_list_bt = vec2list_BT(z_vec, bt_shape, cutoffs_c)
            # 送入原始模块计算一步 f(z)
            out_list = self.func_(z_list_bt, x_list)  # 每个是 (B, C_i, T)
            # 再打包回 (B*T, sumC, 1)
            out_vec, _, _ = list2vec_BT(out_list)
            return out_vec

        jac_loss = torch.tensor(0.0).to(device)
        with torch.no_grad():
            result = self.f_solver(func, z1, threshold=self.f_thres, stop_mode=self.stop_mode, eps=self.stop_val)
            z1 = result["result"]
        new_z1 = z1
        if self.training:
            new_z1 = func(z1.requires_grad_())
            if compute_jac_loss:
                jac_loss = jac_loss_estimate(new_z1, z1)
            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                new_grad = self.b_solver(lambda y: torch.autograd.grad(new_z1, z1, y, retain_graph=True)[0] + grad, torch.zeros_like(grad), threshold=self.b_thres, eps=self.stop_val)["result"]
                return new_grad

            self.hook = new_z1.register_hook(backward_hook)
        # net = vec2list(new_z1, cutoffs)
        net = vec2list_BT(new_z1, bt_shape, cutoffs_c)
        return net, jac_loss.view(1, -1), result

    def forward(self, features):
        if not (isinstance(features, list) or isinstance(features, tuple)) or not all(isinstance(i, torch.Tensor) for i in features):
            raise ValueError("Input should be a list or tuple of tensors.")
        if len(features) != self.func_.num_branches:
            raise ValueError("The number of inputs should be equal to the number of branches.")
        deq_fusion, jacbian_loss, trace = self.featureFusion(features)
        return deq_fusion, jacbian_loss, trace


# class DEQEQFusionBlock(nn.Module):
#     # DEQEQFusionBlock
#     def __init__(
#         self,
#         fusion_channels=[50, 50],
#         kernel_size=3,
#         bias=False,
#         stop_mod="abs",  # "rel"
#         stop_val=1e-5,
#         fsolver="anderson",
#         fthresh=105,  # 35
#         fblock_type="smallgatedblock",
#         bsolver="anderson",
#         bthresh=106,  # 36
#         bblock_type="deqfusionblock",
#     ):
#         """
#         DEQEQFusionBlock is a fusion block that uses DEQ to fuse features from different modalities.
#         Arguments:
#             fusion_channels: List of channels for each modality, last one is the fusion channel.
#             kernel_size: Kernel size for the convolutional layers.
#             bias: Whether to use bias in the convolutional layers.
#             stop_mod: Stopping mode for DEQ solver.
#             stop_val: Stopping value for DEQ solver.
#             fsolver: Solver type for modality solver.
#             fthresh: Threshold for modality solver.
#             fblock_type: Type of block used in the modality solver.
#             bsolver: Solver type for fusion solver.
#             bthresh: Threshold for fusion solver.
#             bblock_type: Type of block used in the fusion solver.
#         """
#         super(DEQEQFusionBlock, self).__init__()
#         self.func_ = DEQEQFusionModule(
#             num_out_dims=fusion_channels,
#             kernel_size=kernel_size,
#             bias=bias,
#             fblock_type=fblock_type.lower() if isinstance(fblock_type, str) else fblock_type,
#             bblock_type=bblock_type.lower() if isinstance(bblock_type, str) else bblock_type,
#         )
#         self.f_thres = fthresh
#         self.b_thres = bthresh
#         self.stop_mode = stop_mod
#         self.stop_val = stop_val
#         self.f_solver = SOLVERDICT.get(fsolver.lower()) if isinstance(fsolver, str) else fsolver
#         self.b_solver = SOLVERDICT.get(bsolver.lower()) if isinstance(bsolver, str) else bsolver
#         if self.f_solver is None:
#             Warning(f"Unknown f_solver: {fsolver}, using default solver.")
#             self.f_solver = SOLVERDICT["anderson"]
#         if self.b_solver is None:
#             Warning(f"Unknown b_solver: {bsolver}, using default solver.")
#             self.b_solver = SOLVERDICT["anderson"]
#         self.hook = None

#     def featureFusion(self, features):
#         device = features[0].device
#         x_list = list(features)
#         z_list = [torch.zeros(feature.shape, device=device) for feature in x_list]
#         min_diff = float("inf")
#         out = None
#         idx = -1 
#         for i in range(self.f_thres):
#             z_new = self.func_(z_list, x_list)
#             diff = (z_new[-1] - z_list[-1]).mean().item()
#             if diff < min_diff:
#                 diff = min_diff 
#                 out = z_new
#                 idx = i
#             z_list = z_new
#         return out, idx, min_diff
        
        

#     def forward(self, features):
#         if not (isinstance(features, list) or isinstance(features, tuple)) or not all(isinstance(i, torch.Tensor) for i in features):
#             raise ValueError("Input should be a list or tuple of tensors.")
#         if len(features) != self.func_.num_branches:
#             raise ValueError("The number of inputs should be equal to the number of branches.")
#         deq_fusion, idx, min_diff = self.featureFusion(features)
#         return deq_fusion, idx, min_diff
