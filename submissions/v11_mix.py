import torch
from task import input_t, output_t

# (technique tracked in PROGRESS.md, not the filename)
# v11 = mixed-precision FEASIBILITY probe: blocked-WY (eager, NO streams/graphs) with the
# two big trailing-update GEMMs in BF16 (fp32 accumulate), panel + Gram + T in FP32.
# Question: does selective BF16 on only the trailing GEMM keep correctness on the stress
# cases (band/rowscale/clustered) under the gate? (global TF32 failed band 27.9, rowscale 26.1.)
# Dispatch widened to 128<=n<=1024 so the n512 b16 band/rowscale tests exercise our path.

_BLOCK = 64
_LOWP = torch.bfloat16


def _householder_step(H, tau_all, col, col_end):
    n = H.shape[1]
    x = H[:, col:, col].clone()
    m = x.shape[-1]
    if m <= 1:
        tau_all[:, col] = 0.0
        return
    alpha = x[:, 0]
    norm_x = x.norm(dim=-1)
    sign_a = alpha.sign()
    sign_a[sign_a == 0] = 1.0
    beta = -sign_a * norm_x
    v0 = alpha - beta
    v_norm_sq = v0 ** 2 + (x[:, 1:] ** 2).sum(-1)
    tau_j = torch.where(v_norm_sq > 0.0, 2.0 * v0 ** 2 / v_norm_sq, torch.zeros_like(v_norm_sq))
    safe_v0 = v0.clone()
    safe_v0[safe_v0.abs() < 1e-30] = 1.0
    u_sub = x[:, 1:] / safe_v0.unsqueeze(-1)
    end = min(col_end, n)
    if col + 1 < end:
        T = H[:, col:, col + 1:end]
        uTT = T[:, 0, :] + torch.bmm(u_sub.unsqueeze(1), T[:, 1:, :]).squeeze(1)
        H[:, col, col + 1:end] -= tau_j.unsqueeze(-1) * uTT
        H[:, col + 1:, col + 1:end] -= (
            tau_j.unsqueeze(-1).unsqueeze(-1) * u_sub.unsqueeze(-1) * uTT.unsqueeze(-2)
        )
    H[:, col, col] = beta
    if col + 1 < n:
        H[:, col + 1:, col] = u_sub
    tau_all[:, col] = tau_j


def _build_wy(H, tau_all, k, b, n):
    batch = H.shape[0]
    device = H.device
    dtype = H.dtype
    Y = torch.tril(H[:, k:, k:k + b])
    idx = torch.arange(b, device=device)
    Y[:, idx, idx] = 1.0
    G = torch.bmm(Y.transpose(-1, -2), Y)              # FP32 Gram (sensitive)
    T = torch.zeros(batch, b, b, device=device, dtype=dtype)
    for j in range(b):
        T[:, j, j] = tau_all[:, k + j]
        if j > 0:
            tj = tau_all[:, k + j].view(batch, 1, 1)
            Gj = G[:, :j, j:j + 1]
            T[:, :j, j:j + 1] = -tj * torch.bmm(T[:, :j, :j], Gj)
    return Y, T


def _trailing_update(H, Y, T, k, b):
    # Big GEMMs in BF16 (tensor-core, fp32 accumulate); T-multiply in FP32.
    A_trail = H[:, k:, k + b:]
    Yb = Y.to(_LOWP)
    C = torch.bmm(Yb.transpose(-1, -2), A_trail.to(_LOWP))   # (b, p) bf16
    TC = torch.bmm(T.transpose(-1, -2), C.float())           # fp32
    H[:, k:, k + b:] -= torch.bmm(Yb, TC.to(_LOWP)).float()


def _blocked_wy(data):
    batch, n, _ = data.shape
    H = data.clone()
    tau_all = torch.zeros(batch, n, device=data.device, dtype=data.dtype)
    B = _BLOCK
    for k in range(0, n, B):
        b = min(B, n - k)
        k_end = k + b
        for j in range(b):
            _householder_step(H, tau_all, k + j, k_end)
        if k_end < n:
            Y, T = _build_wy(H, tau_all, k=k, b=b, n=n)
            _trailing_update(H, Y, T, k=k, b=b)
    return H, tau_all


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape
    if 128 <= n <= 1024:
        return _blocked_wy(data)
    return torch.geqrf(data)
