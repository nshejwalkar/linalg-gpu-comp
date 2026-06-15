"""
stage1_type_probe5.py — read mma.py lines 280-700, find _verify_fragment_A for MmaF16BF16Op.

Also: look at atom.py make_mma_atom to see how it produces an MmaAtom,
and what MmaAtom.partition_A / make_fragment_A look like.

Also: try `get_s2t_smem_desc_tensor` approach for getting smem_desc Tensor.
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-type-probe5")


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def type_probe5():
    import inspect
    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.nvgpu.tcgen05 as tc

    base = "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages/cutlass"

    print("=" * 72)
    print("TYPE PROBE 5 — mma.py _verify_fragment_A, atom.py MmaAtom")
    print("=" * 72)

    # 1. Read mma.py lines 280-800 (where MmaOp._verify_fragment_A lives + MmaF16BF16Op)
    print("\n=== mma.py lines 280-800 ===")
    with open(base + "/cute/nvgpu/tcgen05/mma.py") as f:
        mma_src = f.read()
    mma_lines = mma_src.split('\n')
    for i, line in enumerate(mma_lines[280:800], start=281):
        print(f"  {i}: {line}")

    # 2. Read atom.py make_mma_atom + MmaAtom class fully
    print("\n=== atom.py: make_mma_atom + MmaAtom class (search) ===")
    with open(base + "/cute/atom.py") as f:
        atom_src = f.read()
    atom_lines = atom_src.split('\n')
    # Find make_mma_atom
    in_fn = False
    fn_lines = []
    for i, line in enumerate(atom_lines):
        if 'def make_mma_atom' in line or ('class MmaAtom' in line):
            in_fn = True
            fn_lines = [f"  {i+1}: {line}"]
        elif in_fn:
            fn_lines.append(f"  {i+1}: {line}")
            # Check for next top-level def/class that ends this block
            if len(fn_lines) > 2 and line and not line.startswith(' ') and not line.startswith('\t') and line.strip() and not line.startswith('#'):
                in_fn = False
                print('\n'.join(fn_lines[:-1]))
                fn_lines = []
            elif len(fn_lines) > 120:
                in_fn = False
                print('\n'.join(fn_lines))
                fn_lines = []
    if fn_lines:
        print('\n'.join(fn_lines))

    # 3. What is exported from tcgen05.helpers?
    print("\n=== tc.get_s2t_smem_desc_tensor exists? ===")
    fn = getattr(tc, 'get_s2t_smem_desc_tensor', None)
    print(f"  tc.get_s2t_smem_desc_tensor: {fn}")

    # 4. Check what _cute_nvgpu_ir exports
    print("\n=== _cute_nvgpu_ir module exports ===")
    import cutlass._mlir.dialects.cute_nvgpu as nvgpu_ir
    attrs = [a for a in dir(nvgpu_ir) if not a.startswith('_')]
    print(f"  attrs: {attrs}")
    # Find SmemDescType and make_umma_smem_desc
    for name in attrs:
        if 'smem_desc' in name.lower() or 'umma' in name.lower() or 'fragment' in name.lower():
            print(f"  {name}: {getattr(nvgpu_ir, name)}")


@app.local_entrypoint()
def main():
    type_probe5.remote()
