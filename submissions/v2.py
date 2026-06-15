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


def _blocked_wy_into(H, tau_all):
    """Factor H in-place into (R+vectors); fill tau_all. Both are pre-allocated."""
    n = H.shape[1]
    B = _BLOCK
    for k in range(0, n, B):
        b = min(B, n - k)
        k_end = k + b
        for j in range(b):
            _householder_step(H, tau_all, k + j, k_end)
        if k_end < n:
            Y, T = _build_wy(H, tau_all, k=k, b=b, n=n)
            _trailing_update(H, Y, T, k=k, b=b)


# ── CUDA-graph wrapper: capture the (static-shape) blocked-WY call once per
#    (batch, n, dtype), then replay. Replay issues the whole kernel sequence with
#    one cudaGraphLaunch, removing the ~CPU-dispatch overhead that dominates this
#    launch-bound workload (and makes timing CPU-speed-independent).

_graph_cache = {}


def _get_graph(batch, n, device, dtype):
    static_in = torch.empty(batch, n, n, device=device, dtype=dtype)
    static_H = static_in.clone()
    static_tau = torch.zeros(batch, n, device=device, dtype=dtype)

    # Warmup on a side stream (required before capture).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_H.copy_(static_in)
            static_tau.zero_()
            _blocked_wy_into(static_H, static_tau)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_H.copy_(static_in)
        static_tau.zero_()
        _blocked_wy_into(static_H, static_tau)
    return g, static_in, static_H, static_tau


def _graphed_qr(data):
    key = (data.shape[0], data.shape[1], data.dtype)
    entry = _graph_cache.get(key)
    if entry is None:
        entry = _get_graph(key[0], key[1], data.device, data.dtype)
        _graph_cache[key] = entry
    g, static_in, static_H, static_tau = entry
    static_in.copy_(data)
    g.replay()
    return static_H.clone(), static_tau.clone()


def _ours_wins(batch: int, n: int) -> bool:
    # Broad regime for MEASURING graphed perf; tune in v3 from the numbers.
    return n >= 256


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape
    if not _ours_wins(batch, n):
        return torch.geqrf(data)
    try:
        return _graphed_qr(data)
    except Exception:
        # Fallback: non-graphed in-place factor (e.g. capture unsupported).
        H = data.clone()
        tau_all = torch.zeros(batch, n, device=data.device, dtype=data.dtype)
        _blocked_wy_into(H, tau_all)
        return H, tau_all
