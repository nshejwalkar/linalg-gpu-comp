import torch
from task import input_t, output_t

# (technique tracked in PROGRESS.md, not the filename)

_BLOCK = 64


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
    G = torch.bmm(Y.transpose(-1, -2), Y)
    T = torch.zeros(batch, b, b, device=device, dtype=dtype)
    for j in range(b):
        T[:, j, j] = tau_all[:, k + j]
        if j > 0:
            tj = tau_all[:, k + j].view(batch, 1, 1)
            Gj = G[:, :j, j:j + 1]
            T[:, :j, j:j + 1] = -tj * torch.bmm(T[:, :j, :j], Gj)
    return Y, T


def _trailing_update(H, Y, T, k, b):
    A_trail = H[:, k:, k + b:]
    C = torch.bmm(Y.transpose(-1, -2), A_trail)
    TC = torch.bmm(T.transpose(-1, -2), C)
    H[:, k:, k + b:] -= torch.bmm(Y, TC)


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


def _ours_wins(batch: int, n: int) -> bool:
    # blocked_wy is CPU-dispatch-bound, so its runtime varies with the container's
    # CPU speed (measured b640 n512: 6.7-9.5x; b60 n1024: 0.75-1.14x, flips to a LOSS
    # on a slow-CPU run). Dispatch ONLY to the robustly-huge win (very large batch),
    # send everything else to cuSOLVER. CUDA graphs (v2) will remove the CPU
    # dependence and let us widen this.
    return batch >= 128 and n >= 256


def custom_kernel(data: input_t) -> output_t:
    """Shape-dispatch: blocked WY where it beats cuSOLVER, else torch.geqrf."""
    batch, n, _ = data.shape
    if _ours_wins(batch, n):
        return _blocked_wy(data)
    return torch.geqrf(data)
