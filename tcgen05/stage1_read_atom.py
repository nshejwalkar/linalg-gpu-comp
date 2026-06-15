"""Read atom.py and check mma_atom_call + make_umma_smem_desc return type."""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-read-atom")


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def read_atom():
    import os, inspect

    base = "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages/cutlass"

    # Read atom.py - has mma_atom_call
    atom_path = os.path.join(base, "cute/atom.py")
    print(f"\n=== {atom_path} ===")
    with open(atom_path) as f:
        content = f.read()
    # Print sections around mma_atom_call and _normalize_variadic
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'mma_atom_call' in line or '_normalize_variadic' in line or 'smem_desc' in line:
            start = max(0, i-2)
            end = min(len(lines), i+10)
            print(f"  line {i+1}: {line}")

    # Also find mma_atom_call function
    in_fn = False
    fn_lines = []
    for i, line in enumerate(lines):
        if 'def mma_atom_call' in line or 'def _normalize_variadic' in line:
            in_fn = True
            fn_lines = [f"  {i+1}: {line}"]
        elif in_fn:
            fn_lines.append(f"  {i+1}: {line}")
            if len(fn_lines) > 30:
                in_fn = False
                print('\n'.join(fn_lines))
                fn_lines = []
    if fn_lines:
        print('\n'.join(fn_lines))

    # Check what make_umma_smem_desc returns
    print("\n=== Testing make_umma_smem_desc return type ===")
    # We can't test inside a @cute.kernel here, but we can inspect the signature
    import cutlass.cute.nvgpu.tcgen05 as tc
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import BFloat16, Float32, Uint32
    from cutlass.utils import SmemAllocator
    import inspect as ins
    fn = getattr(tc, "make_umma_smem_desc", None)
    if fn:
        print(f"  make_umma_smem_desc{ins.signature(fn)}")
        print(f"  return annotation: {ins.signature(fn).return_annotation}")

    # Find the Tensor type check
    print("\n=== _normalize_variadic_tensor_operand ===")
    fn2 = getattr(cute, "_normalize_variadic_tensor_operand", None)
    if fn2:
        print(f"  found in cute: {fn2}")
    else:
        from cutlass.cute.atom import _normalize_variadic_tensor_operand
        print(f"  source:\n{ins.getsource(_normalize_variadic_tensor_operand)}")


@app.local_entrypoint()
def main():
    read_atom.remote()
