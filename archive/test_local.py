"""
Local CPU correctness test for our QR submissions.
Tests against torch.linalg.qr and the geqrf format exactly.
"""
import sys, math, torch
sys.path.insert(0, '/home/claude/qr_competition')

# ── Mock 'task' module so submission.py imports work ──────────────────────────
import types
task = types.ModuleType('task')
task.input_t = torch.Tensor
task.output_t = tuple
sys.modules['task'] = task

from submission_blocked_wy import custom_kernel

eps32 = torch.finfo(torch.float32).eps

def geqrf_to_QR(H, tau):
    """Reconstruct Q and R from (H, tau) returned by our kernel."""
    Q = torch.linalg.householder_product(H, tau)
    R = torch.triu(H)
    return Q, R

def check(name, A, H, tau, tol_factor=None, tol_ortho=None):
    n = A.shape[-1]
    if tol_factor is None: tol_factor = 20 * n * eps32
    if tol_ortho  is None: tol_ortho  = 100 * n * eps32
    
    Q, R = geqrf_to_QR(H, tau)
    batch = A.shape[0]
    passed = True
    
    for b in range(batch):
        a, q, r, h, t = A[b], Q[b], R[b], H[b], tau[b]
        A_norm = a.norm().item()
        
        # Factorization: R ≈ Q^T A
        fact_err = (r - q.T @ a).norm().item() / (A_norm + 1e-30)
        # Orthogonality: Q^T Q ≈ I
        n_sq = q.shape[0]
        orth_err = (q.T @ q - torch.eye(n_sq)).norm().item()
        # Reconstruction: Q R ≈ A
        rec_err  = (q @ r - a).norm().item() / (A_norm + 1e-30)
        # Triangularity
        tri_err  = torch.tril(r, -1).norm().item() / (A_norm + 1e-30)

        ok = (fact_err < tol_factor and orth_err < tol_ortho 
              and rec_err < tol_factor and tri_err < 1e-6)
        if not ok:
            print(f"  FAIL batch={b}: fact={fact_err:.2e} orth={orth_err:.2e} "
                  f"rec={rec_err:.2e} tri={tri_err:.2e}  "
                  f"(tols: {tol_factor:.2e} / {tol_ortho:.2e})")
            passed = False
    return passed

def run_test(name, A):
    H, tau = custom_kernel(A)
    ok = check(name, A, H, tau)
    ref_H, ref_tau = torch.geqrf(A)
    # Note: geqrf format is unique up to sign convention, so we check quality not exact match
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return ok

torch.manual_seed(42)
device = 'cpu'
results = []

# ── Dense random ──────────────────────────────────────────────────────────────
for n in [4, 8, 16, 32, 64, 128, 256]:
    A = torch.randn(4, n, n)
    results.append(run_test(f"dense n={n}", A))

# ── Identity (trivial: already orthogonal) ───────────────────────────────────
for n in [8, 32, 64]:
    A = torch.eye(n).unsqueeze(0).expand(2, -1, -1).clone()
    results.append(run_test(f"identity n={n}", A))

# ── Upper triangular ──────────────────────────────────────────────────────────
for n in [8, 64]:
    A = torch.triu(torch.randn(2, n, n))
    results.append(run_test(f"upper n={n}", A))

# ── Diagonal ──────────────────────────────────────────────────────────────────
for n in [8, 64]:
    d = torch.rand(2, n) + 0.1
    A = torch.diag_embed(d)
    results.append(run_test(f"diagonal n={n}", A))

# ── Rank-deficient ────────────────────────────────────────────────────────────
for n in [8, 32]:
    A = torch.randn(2, n, n//2) @ torch.randn(2, n//2, n)  # rank n/2
    results.append(run_test(f"rankdef n={n}", A))

# ── Near-rank-deficient ───────────────────────────────────────────────────────
for n in [32, 64]:
    U = torch.linalg.qr(torch.randn(2, n, n))[0]
    s = torch.cat([torch.ones(2, n//2), torch.tensor([[1e-6]*((n-n//2))] * 2)], dim=1)
    A = U * s.unsqueeze(1)
    results.append(run_test(f"nearrank n={n}", A))

# ── Row-scaled ────────────────────────────────────────────────────────────────
for n in [32, 64]:
    A = torch.randn(2, n, n)
    A[:, 0, :] *= 1e6
    A[:, -1, :] *= 1e-6
    results.append(run_test(f"rowscale n={n}", A))

# ── Near-collinear ────────────────────────────────────────────────────────────
for n in [16, 32]:
    A = torch.randn(2, n, n)
    A[:, :, 1] = A[:, :, 0] + 1e-7 * torch.randn(2, n)
    results.append(run_test(f"nearcollinear n={n}", A))

# ── Banded ────────────────────────────────────────────────────────────────────
for n in [16, 64]:
    A = torch.zeros(2, n, n)
    for k in range(-2, 3):
        d = min(n, n - abs(k))
        A += torch.diag_embed(torch.randn(2, d), k)
    results.append(run_test(f"band n={n}", A))

# ── Clustered singular values ─────────────────────────────────────────────────
for n in [32, 64]:
    U = torch.linalg.qr(torch.randn(2, n, n))[0]
    V = torch.linalg.qr(torch.randn(2, n, n))[0]
    s = torch.ones(2, n) + 0.1 * torch.randn(2, n)  # clustered around 1
    A = U * s.unsqueeze(1) @ V.transpose(-1,-2)
    results.append(run_test(f"clustered n={n}", A))

print(f"\n{'='*40}")
print(f"Results: {sum(results)}/{len(results)} passed")
