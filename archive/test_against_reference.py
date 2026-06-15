"""
Validate submission_blocked_wy.py against the ACTUAL gpu-mode reference.py
generate_input() and check_implementation(), using the real task.yml test
specs (CPU-feasible subset: n <= 512).
"""
import sys, torch, yaml

sys.path.insert(0, '/home/claude/qr_competition')
sys.path.insert(0, '/home/claude/qr_competition/reference_repo')

from reference import generate_input, check_implementation
from submission_blocked_wy import custom_kernel

with open('/home/claude/qr_competition/reference_repo/task.yml') as f:
    spec = yaml.safe_load(f)

results = []
for t in spec['tests']:
    if t['n'] > 512:
        print(f"  SKIP  n={t['n']} (too slow on CPU)")
        continue
    A = generate_input(t['batch'], t['n'], t['cond'], t['seed'], t.get('case', 'dense'))
    H, tau = custom_kernel(A)
    ok, msg = check_implementation(A, (H, tau))
    case = t.get('case', 'dense')
    status = 'PASS' if ok else 'FAIL'
    print(f"  {status}  batch={t['batch']:>3} n={t['n']:>4} cond={t['cond']} case={case:<12}  {msg}")
    results.append(ok)

print(f"\n{sum(results)}/{len(results)} passed")
