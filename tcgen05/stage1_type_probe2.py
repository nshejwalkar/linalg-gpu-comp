"""
stage1_type_probe2.py — probe TensorSSA, SmemDescType, and make_tensor(ir.Value).

Key findings from probe1:
- cute.TensorSSA exists (cutlass.cute.tensor.TensorSSA)
- cute.make_tensor accepts ir.Value as iterator arg
- make_umma_smem_desc returns SmemDescType (C extension result)
- algorithm.py has no 'desc' lines (weird - check again)

New hypotheses to test:
A) Does make_umma_smem_desc return a TensorSSA? (Subclass of Tensor)
B) Can cute.make_tensor(desc_ir_val, layout) create a valid smem_desc Tensor?
C) Does TensorSSA have a constructor that accepts ir.Value?
D) Read tcgen05/__init__.py to find the actual return of make_umma_smem_desc
E) Read algorithm.py gemm function in full to see smem_desc handling
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-type-probe2")


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def type_probe2():
    import inspect
    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.nvgpu.tcgen05 as tc
    from cutlass.cutlass_dsl import BFloat16, Float32, Uint32
    from cutlass.cute.nvgpu.common import OperandMajorMode

    print("=" * 72)
    print("TYPE PROBE 2 — TensorSSA, SmemDesc, make_tensor(ir.Value)")
    print("=" * 72)

    # 1. Inspect TensorSSA
    print("\n=== cute.TensorSSA ===")
    from cutlass.cute.tensor import TensorSSA
    print(f"  TensorSSA MRO: {[c.__name__ for c in TensorSSA.__mro__]}")
    print(f"  TensorSSA is subclass of cute.Tensor: {issubclass(TensorSSA, cute.Tensor)}")
    try:
        src = inspect.getsource(TensorSSA.__init__)
        print(f"  TensorSSA.__init__ source:\n{src[:800]}")
    except Exception as e:
        print(f"  TensorSSA.__init__: {e}")
    # Show TensorSSA methods
    print(f"  TensorSSA attrs: {[a for a in dir(TensorSSA) if not a.startswith('__')]}")

    # 2. Read tcgen05/__init__.py to find make_umma_smem_desc return
    print("\n=== tcgen05/__init__.py make_umma_smem_desc full source ===")
    base = "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages/cutlass"
    tc_path = base + "/cute/nvgpu/tcgen05/__init__.py"
    with open(tc_path) as f:
        content = f.read()
    lines = content.split('\n')
    in_fn = False
    fn_lines = []
    for i, line in enumerate(lines):
        if 'def make_umma_smem_desc' in line:
            in_fn = True
            fn_lines = []
        if in_fn:
            fn_lines.append(f"  {i+1}: {line}")
            # Stop at next def or after 60 lines
            if len(fn_lines) > 1 and 'def ' in line and 'make_umma_smem_desc' not in line:
                in_fn = False
                break
            if len(fn_lines) > 60:
                in_fn = False
                break
    print('\n'.join(fn_lines))

    # 3. Read algorithm.py gemm function in full
    print("\n=== algorithm.py - gemm function ===")
    alg_path = base + "/cute/algorithm.py"
    with open(alg_path) as f:
        content2 = f.read()
    lines2 = content2.split('\n')
    in_fn = False
    fn_lines = []
    for i, line in enumerate(lines2):
        if 'def gemm(' in line or 'def gemm (' in line:
            in_fn = True
            fn_lines = []
        if in_fn:
            fn_lines.append(f"  {i+1}: {line}")
            if len(fn_lines) > 1 and 'def ' in line and 'def gemm' not in line:
                in_fn = False
                break
            if len(fn_lines) > 80:
                in_fn = False
                break
    print('\n'.join(fn_lines))

    # 4. Search algorithm.py for smem_desc references
    print("\n=== algorithm.py full text (lines with 'smem' or 'desc' or 'Tensor') ===")
    for i, line in enumerate(lines2):
        if 'smem' in line.lower() or 'desc' in line.lower() or 'TensorSSA' in line:
            print(f"  {i+1}: {line}")

    # 5. Look for _cute_nvgpu_ir alternatives
    print("\n=== where does tcgen05 import _cute_nvgpu_ir from? ===")
    for i, line in enumerate(lines):
        if 'import' in line or '_cute' in line:
            print(f"  {i+1}: {line}")
        if i > 30:
            break

    # 6. Full tcgen05/__init__.py imports + first 50 lines
    print("\n=== tcgen05/__init__.py first 80 lines ===")
    for i, line in enumerate(lines[:80]):
        print(f"  {i+1}: {line}")


@app.local_entrypoint()
def main():
    type_probe2.remote()
