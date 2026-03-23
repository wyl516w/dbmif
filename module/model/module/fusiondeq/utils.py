""" 
    DEQ fusion module
    Modified based on the DEQ repo (https://github.com/locuslab/deq)
"""
import torch.nn as nn
import torch

# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

def list2vec(z1_list):
    """Convert list of tensors to a vector"""
    bsz = z1_list[0].size(0)
    return torch.cat([elem.reshape(bsz, -1, 1) for elem in z1_list], dim=1)


def vec2list(z1, cutoffs):
    """Convert a vector back to a list, via the cutoffs specified"""
    bsz = z1.shape[0]
    z1_list = []
    start_idx, end_idx = 0, cutoffs[0][0] * cutoffs[0][1]
    for i in range(len(cutoffs)):
        z1_list.append(z1[:, start_idx:end_idx].view(bsz, *cutoffs[i]))
        if i < len(cutoffs) - 1:
            start_idx = end_idx
            end_idx += cutoffs[i + 1][0] * cutoffs[i + 1][1]
    return z1_list

def list2vec_BT(z_list):
    """
    输入: z_list = [z_i], 每个 z_i 形状 (B, C_i, T)
    输出:
      z_vec: (B*T, sum(C_i), 1)
      bt_shape: (B, T)
      cutoffs_c: [C_1, C_2, ...] 仅记录各分支的通道数
    """
    assert isinstance(z_list, (list, tuple)) and len(z_list) > 0
    B, _, T = z_list[0].shape
    cutoffs_c = []
    chunks = []
    for zi in z_list:
        assert zi.shape[0] == B and zi.shape[2] == T, "各分支需同 B、T"
        Ci = zi.shape[1]
        cutoffs_c.append(Ci)
        # (B, C, T) -> (B, T, C) -> (B*T, C, 1)
        chunks.append(zi.permute(0, 2, 1).contiguous().view(B*T, Ci, 1))
    z_vec = torch.cat(chunks, dim=1)  # (B*T, sumC, 1)
    return z_vec, (B, T), cutoffs_c


def vec2list_BT(z_vec, bt_shape, cutoffs_c):
    """
    输入:
      z_vec: (B*T, sumC, 1)
      bt_shape: (B, T)
      cutoffs_c: [C_1, C_2, ...]
    输出:
      z_list: [ (B, C_i, T) ... ]
    """
    B, T = bt_shape
    assert z_vec.dim() == 3 and z_vec.shape[0] == B*T
    z_list = []
    start = 0
    for Ci in cutoffs_c:
        seg = z_vec[:, start:start+Ci, :]                # (B*T, C_i, 1)
        seg = seg.view(B, T, Ci).permute(0, 2, 1).contiguous()  # -> (B, C_i, T)
        z_list.append(seg)
        start += Ci
    return z_list
