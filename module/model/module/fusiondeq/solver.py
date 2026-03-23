# Modified based on the DEQ repo.

import torch
import numpy as np
import pickle


def _safe_norm(v):
    if not torch.isfinite(v).all():
        return torch.tensor(float("inf")).to(v)
    return torch.norm(v)


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    ite = 0
    phi_a0 = phi(alpha0)  # First do an update with step size 1
    if phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        return alpha0, phi_a0, ite

    # Otherwise, compute the minimizer of a quadratic interpolant
    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    while alpha1 > amin:  # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1 - alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0 * alpha1) - alpha1**2 * (
            phi_a0 - phi0 - derphi0 * alpha0
        )
        a = a / factor
        b = -(alpha0**3) * (phi_a1 - phi0 - derphi0 * alpha1) + alpha1**3 * (
            phi_a0 - phi0 - derphi0 * alpha0
        )
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * derphi0))) / (3.0 * a)
        phi_a2 = phi(alpha2)
        ite += 1

        if phi_a2 <= phi0 + c1 * alpha2 * derphi0:
            return alpha2, phi_a2, ite

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2 / alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1, ite


def line_search(update, x0, g0, g, nstep=0, on=True):
    """
    `update` is the propsoed direction of update.
    Code adapted from scipy.
    """
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0) ** 2]
    s_norm = torch.norm(x0) / torch.norm(update)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]  # If the step size is so small... just return something
        x_est = x0 + s * update
        g0_new = g(x_est)
        phi_new = _safe_norm(g0_new) ** 2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new

    if on:
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-2)
    if (not on) or s is None:
        s = 1.0
        ite = 0

    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)
    return x_est, g0_new, x_est - x0, g0_new - g0, ite


def rmatvec(part_Us, part_VTs, x):
    # Compute x^T(-I + UV^T)
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if part_Us.nelement() == 0:
        return -x
    xTU = torch.einsum("bij, bijd -> bd", x, part_Us)  # (N, threshold)
    return -x + torch.einsum(
        "bd, bdij -> bij", xTU, part_VTs
    )  # (N, 2d, L'), but should really be (N, 1, (2d*L'))


def matvec(part_Us, part_VTs, x):
    # Compute (-I + UV^T)x
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if part_Us.nelement() == 0:
        return -x
    VTx = torch.einsum("bdij, bij -> bd", part_VTs, x)  # (N, threshold)
    return -x + torch.einsum(
        "bijd, bd -> bij", part_Us, VTx
    )  # (N, 2d, L'), but should really be (N, (2d*L'), 1)


def broyden(f, x0, threshold, eps=1e-3, stop_mode="rel", ls=False, name="unknown"):
    bsz, total_hsize, seq_len = x0.size()
    g = lambda y: f(y) - y
    dev = x0.device
    alternative_mode = "rel" if stop_mode == "abs" else "abs"

    x_est = x0  # (bsz, 2d, L')
    gx = g(x_est)  # (bsz, 2d, L')
    nstep = 0
    tnstep = 0

    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(bsz, total_hsize, seq_len, threshold, device=dev)
    # One can also use an L-BFGS scheme to further reduce memory
    VTs = torch.zeros(bsz, threshold, total_hsize, seq_len, device=dev)
    update = -matvec(
        Us[:, :, :, :nstep], VTs[:, :nstep], gx
    )  # Formally should be -torch.matmul(inv_jacobian (-I), gx)
    prot_break = False

    # To be used in protective breaks
    protect_thres = (1e6 if stop_mode == "abs" else 1e3) * seq_len
    new_objective = 1e8

    trace_dict = {"abs": [], "rel": []}
    lowest_dict = {"abs": 1e8, "rel": 1e8}
    lowest_step_dict = {"abs": 0, "rel": 0}
    nstep, lowest_xest, lowest_gx = 0, x_est, gx

    while nstep < threshold:
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g, nstep=nstep, on=ls)
        nstep += 1
        tnstep += ite + 1
        abs_diff = torch.norm(gx).item()
        rel_diff = abs_diff / (torch.norm(gx + x_est).item() + 1e-9)
        diff_dict = {"abs": abs_diff, "rel": rel_diff}
        trace_dict["abs"].append(abs_diff)
        trace_dict["rel"].append(rel_diff)
        for mode in ["rel", "abs"]:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest, lowest_gx = x_est.clone().detach(), gx.clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = nstep

        new_objective = diff_dict[stop_mode]
        if new_objective < eps:
            break
        if (
            new_objective < 3 * eps
            and nstep > 30
            and np.max(trace_dict[stop_mode][-30:]) / np.min(trace_dict[stop_mode][-30:]) < 1.3
        ):
            # if there's hardly been any progress in the last 30 steps
            break
        if new_objective > trace_dict[stop_mode][0] * protect_thres:
            prot_break = True
            break

        part_Us, part_VTs = Us[:, :, :, : nstep - 1], VTs[:, : nstep - 1]
        vT = rmatvec(part_Us, part_VTs, delta_x)
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx)) / torch.einsum(
            "bij, bij -> b", vT, delta_gx
        )[:, None, None]
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:, nstep - 1] = vT
        Us[:, :, :, nstep - 1] = u
        update = -matvec(Us[:, :, :, :nstep], VTs[:, :nstep], gx)

    # Fill everything up to the threshold length
    for _ in range(threshold + 1 - len(trace_dict[stop_mode])):
        trace_dict[stop_mode].append(lowest_dict[stop_mode])
        trace_dict[alternative_mode].append(lowest_dict[alternative_mode])

    return {
        "result": lowest_xest,
        "lowest": lowest_dict[stop_mode],
        "nstep": lowest_step_dict[stop_mode],
        "prot_break": prot_break,
        "abs_trace": trace_dict["abs"],
        "rel_trace": trace_dict["rel"],
        "eps": eps,
        "threshold": threshold,
    }


def anderson(f, x0, m=6, lam=1e-4, threshold=50, eps=1e-3, stop_mode="rel", beta=1.0, **kwargs):
    # def anderson(f, x0, m=6, lam=1e-4, threshold=50, eps=1e-4, stop_mode='rel', beta=1.0, **kwargs):
    """Anderson acceleration for fixed point iteration."""
    bsz, d, L = x0.shape
    alternative_mode = "rel" if stop_mode == "abs" else "abs"
    X = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape_as(x0)).reshape(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    #     trace_dict = {'abs': [],
    #                   'rel': []}
    trace_dict = {
        "abs": [
            (F[:, 0] - X[:, 0]).view_as(x0).norm().item(),
            (F[:, 1] - X[:, 1]).view_as(x0).norm().item(),
        ],
        "rel": [
            (F[:, 0] - X[:, 0]).view_as(x0).norm().item() / (1e-5 + F[:, 0].norm().item()),
            (F[:, 1] - X[:, 1]).view_as(x0).norm().item() / (1e-5 + F[:, 1].norm().item()),
        ],
    }
    lowest_dict = {"abs": 1e8, "rel": 1e8}
    lowest_step_dict = {"abs": 0, "rel": 0}
    lowest_xest, lowest_gx = X[:, 0].view_as(x0).clone().detach(), (F[:, 0] - X[:, 0]).view_as(x0).clone().detach()
    
    for k in range(2, threshold):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1 : n + 1, 1 : n + 1] = (
            torch.bmm(G, G.transpose(1, 2))
            + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        )
        try:
            alpha = torch.linalg.solve(H[:, : n + 1, : n + 1], y[:, : n + 1])
        except:
            alpha = torch.linalg.pinv(H[:, : n + 1, : n + 1]) @ y[:, : n + 1]
        alpha = alpha[:, 1 : n + 1, 0]  # (bsz x n)

        X[:, k % m] = (
            beta * (alpha[:, None] @ F[:, :n])[:, 0]
            + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        )
        F[:, k % m] = f(X[:, k % m].reshape_as(x0)).reshape(bsz, -1)
        gx = (F[:, k % m] - X[:, k % m]).view_as(x0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + F[:, k % m].norm().item())
        diff_dict = {"abs": abs_diff, "rel": rel_diff}
        trace_dict["abs"].append(abs_diff)
        trace_dict["rel"].append(rel_diff)

        for mode in ["rel", "abs"]:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest, lowest_gx = (
                        X[:, k % m].view_as(x0).clone().detach(),
                        gx.clone().detach(),
                    )
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k

        if trace_dict[stop_mode][-1] < eps:
            for _ in range(threshold - 1 - k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break

    out = {
        "result": lowest_xest,
        "lowest": lowest_dict[stop_mode],
        "nstep": lowest_step_dict[stop_mode],
        "prot_break": False,
        "abs_trace": trace_dict["abs"],
        "rel_trace": trace_dict["rel"],
        "eps": eps,
        "threshold": threshold,
    }
    X = F = None
    return out

def analyze_broyden(res_info, err=None, judge=True, name="forward", training=True, save_err=True):
    """
    For debugging use only :-)
    """
    res_est = res_info["result"]
    nstep = res_info["nstep"]
    diff = res_info["diff"]
    diff_detail = res_info["diff_detail"]
    prot_break = res_info["prot_break"]
    trace = res_info["trace"]
    eps = res_info["eps"]
    threshold = res_info["threshold"]
    if judge:
        return nstep >= threshold or (nstep == 0 and (diff != diff or diff > eps)) or prot_break or torch.isnan(res_est).any()

    assert err is not None, "Must provide err information when not in judgment mode"
    prefix, color = ("", "red") if name == "forward" else ("back_", "blue")
    eval_prefix = "" if training else "eval_"

    # Case 1: A nan entry is produced in Broyden
    if torch.isnan(res_est).any():
        msg = f"WARNING: nan found in Broyden's {name} result. Diff: {diff}"
        print(msg)
        if save_err:
            pickle.dump(err, open(f"{prefix}{eval_prefix}nan.pkl", "wb"))
        return (1, msg, res_info)

    # Case 2: Unknown problem with Broyden's method (probably due to nan update(s) to the weights)
    if nstep == 0 and (diff != diff or diff > eps):
        msg = f"WARNING: Bad Broyden's method {name}. Why?? Diff: {diff}. STOP."
        print(msg)
        if save_err:
            pickle.dump(err, open(f"{prefix}{eval_prefix}badbroyden.pkl", "wb"))
        return (2, msg, res_info)

    # Case 3: Protective break during Broyden (so that it does not diverge to infinity)
    if prot_break and np.random.uniform(0, 1) < 0.05:
        msg = (f"WARNING: Hit Protective Break in {name}. Diff: {diff}. Total Iter: {len(trace)}",)
        print(msg)
        if save_err:
            pickle.dump(err, open(f"{prefix}{eval_prefix}prot_break.pkl", "wb"))
        return (3, msg, res_info)

    return (-1, "", res_info)

def weight_tie(f, x0, m=6, eps=1e-3, threshold=50, stop_mode="rel"):
    print("Using weight tie")
    bsz, L = x0.shape
    alternative_mode = "rel" if stop_mode == "abs" else "abs"
    X = torch.zeros(bsz, L, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, L, dtype=x0.dtype, device=x0.device)
    X, F = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)

    trace_dict = {
        "abs": [(F - X).view_as(x0).norm().item()],
        "rel": [(F - X).view_as(x0).norm().item() / (1e-5 + F.norm().item())],
    }
    lowest_dict = {"abs": 1e8, "rel": 1e8}
    lowest_step_dict = {"abs": 0, "rel": 0}

    for k in range(1, threshold):
        X = F.clone()
        F = f(F.reshape_as(x0)).reshape(bsz, -1)
        gx = (F - X).view_as(x0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + F.norm().item())
        #         print(abs_diff, rel_diff)
        diff_dict = {"abs": abs_diff, "rel": rel_diff}
        trace_dict["abs"].append(abs_diff)
        trace_dict["rel"].append(rel_diff)

        for mode in ["rel", "abs"]:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest, lowest_gx = X.view_as(x0).clone().detach(), gx.clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k

    #         if trace_dict[stop_mode][-1] < eps:
    #             for _ in range(threshold-1-k):
    #                 trace_dict[stop_mode].append(lowest_dict[stop_mode])
    #                 trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
    #             break

    out = {
        "result": lowest_xest,
        "lowest": lowest_dict[stop_mode],
        "nstep": lowest_step_dict[stop_mode],
        "prot_break": False,
        "abs_trace": trace_dict["abs"],
        "rel_trace": trace_dict["rel"],
        "threshold": threshold,
    }
    X = F = None
    return out

def anderson_over_channels(f_orig, x0, **anderson_kwargs):
    """
    在 C 维上做 Anderson，加速每个 (b,t) 的不动点迭代。
    x0: (B, C, T)
    f_orig: 输入/输出均为 (B, C, T) 的映射（逐时刻独立或你保证可按帧处理）
    """
    assert x0.dim() == 3, "x0 must be (B,C,T)"
    B, C, T = x0.shape

    # 变换到 (B*T, C, 1)，把 (B,T) 视为 batch
    x0_bt_c1 = x0.permute(0, 2, 1).contiguous().view(B*T, C, 1)

    # 包装 f：保持与 anderson(x).shape=(B*T, C, 1) 一致
    def f_wrap(x_bt_c1):
        # (B*T, C, 1) -> (B, C, T)
        x_bct = x_bt_c1.view(B, T, C).permute(0, 2, 1).contiguous()
        y_bct = f_orig(x_bct)            # 需同形状 (B, C, T)
        # (B, C, T) -> (B*T, C, 1)
        y_bt_c1 = y_bct.permute(0, 2, 1).contiguous().view(B*T, C, 1)
        return y_bt_c1

    # 关键：只在 C 维迭代 -> 传入 anderson 的张量形状 (bsz=B*T, d=C, L=1)
    out = anderson(f_wrap, x0_bt_c1, **anderson_kwargs)

    # 结果还原回 (B, C, T)
    out["result"] = out["result"].view(B, T, C).permute(0, 2, 1).contiguous()
    return out


def broyden_over_channels(f_orig, x0, **broyden_kwargs):
    """
    在 C 维上做 Broyden，加速每个 (b,t) 的不动点迭代。
    x0: (B, C, T)
    f_orig: 输入/输出均为 (B, C, T) 的映射
    透传 broyden 的参数: threshold, eps, stop_mode, ls, name 等
    """
    assert x0.dim() == 3, "x0 must be (B,C,T)"
    B, C, T = x0.shape
    # (B,C,T) -> (B*T, C, 1)
    x0_bt_c1 = x0.permute(0, 2, 1).contiguous().view(B*T, C, 1)

    def f_wrap(x_bt_c1):
        # (B*T, C, 1) -> (B,C,T)
        x_bct = x_bt_c1.view(B, T, C).permute(0, 2, 1).contiguous()
        y_bct = f_orig(x_bct)                        # 仍为 (B,C,T)
        # (B,C,T) -> (B*T, C, 1)
        y_bt_c1 = y_bct.permute(0, 2, 1).contiguous().view(B*T, C, 1)
        return y_bt_c1

    out = broyden(f_wrap, x0_bt_c1, **broyden_kwargs)
    # 结果还原回 (B,C,T)
    out["result"] = out["result"].view(B, T, C).permute(0, 2, 1).contiguous()
    return out


def weight_tie_over_channels(f_orig, x0, **wt_kwargs):
    """
    在 C 维上做 weight_tie（朴素不动点迭代），每个 (b,t) 独立。
    x0: (B, C, T)
    f_orig: 输入/输出均为 (B, C, T) 的映射
    透传 weight_tie 的参数: m, eps, threshold, stop_mode 等
    """
    assert x0.dim() == 3, "x0 must be (B,C,T)"
    B, C, T = x0.shape
    # (B,C,T) -> (B*T, C)
    x0_bt_c = x0.permute(0, 2, 1).contiguous().view(B*T, C)

    def f_wrap(x_bt_c):
        # (B*T, C) -> (B,C,T)
        x_bct = x_bt_c.view(B, T, C).permute(0, 2, 1).contiguous()
        y_bct = f_orig(x_bct)                        # (B,C,T)
        # (B,C,T) -> (B*T, C)
        y_bt_c = y_bct.permute(0, 2, 1).contiguous().view(B*T, C)
        return y_bt_c

    out = weight_tie(f_wrap, x0_bt_c, **wt_kwargs)
    # 结果还原回 (B,C,T)
    out["result"] = out["result"].view(B, T, C).permute(0, 2, 1).contiguous()
    return out


SOLVERDICT = {
    "broyden": broyden_over_channels,
    "anderson": anderson_over_channels,
    "weight_tie": weight_tie_over_channels,
}


