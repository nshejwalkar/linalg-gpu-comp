"""
stage1_type_probe3.py — read helpers.py, mma.py, find make_fragment_A / make_umma_desc_tensor.

Key: make_umma_smem_desc is in helpers.py. The MmaAtom might have make_fragment_A.
Also: MmaF16BF16Op might have a method that returns descriptor Tensors.
Also: check if cute.make_tensor(smem_desc_ir_val, layout) is valid.
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-type-probe3")


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def type_probe3():
    import inspect
    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.nvgpu.tcgen05 as tc
    from cutlass.cutlass_dsl import BFloat16, Float32, Uint32
    from cutlass.cute.nvgpu.common import OperandMajorMode

    base = "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages/cutlass"

    print("=" * 72)
    print("TYPE PROBE 3 — helpers.py, mma.py, fragment methods")
    print("=" * 72)

    # 1. Read helpers.py fully
    print("\n=== helpers.py FULL ===")
    with open(base + "/cute/nvgpu/tcgen05/helpers.py") as f:
        helpers_src = f.read()
    print(helpers_src[:5000])

    # 2. Check MmaAtom methods
    print("\n=== MmaAtom methods ===")
    mma_op = tc.MmaF16BF16Op(
        BFloat16, Float32, (128, 256, 16),
        tc.CtaGroup.ONE, tc.OperandSource.SMEM,
        OperandMajorMode.K, OperandMajorMode.K,
    )
    mma_atom = cute.make_mma_atom(mma_op)
    print(f"  type(mma_atom): {type(mma_atom)}")
    print(f"  mma_atom attrs: {[a for a in dir(mma_atom) if not a.startswith('_')]}")
    # Try make_fragment_A / make_fragment_B
    for method in ['make_fragment_A', 'make_fragment_B', 'make_fragment_C',
                   'make_desc_A', 'make_desc_B', 'partition_A', 'partition_B',
                   'get_slice', 'accumulate_']:
        val = getattr(mma_atom, method, None)
        if val is not None:
            print(f"  mma_atom.{method}: {val}")
            try:
                print(f"    sig: {inspect.signature(val)}")
            except:
                pass

    # 3. Read mma.py to understand MmaAtom
    print("\n=== mma.py sections (MmaAtom, make_mma_atom) ===")
    with open(base + "/cute/nvgpu/tcgen05/mma.py") as f:
        mma_src = f.read()
    lines = mma_src.split('\n')
    # Find class MmaAtom or relevant parts
    in_class = False
    class_lines = []
    for i, line in enumerate(lines):
        if 'class MmaAtom' in line or 'def make_mma_atom' in line or 'make_umma' in line:
            in_class = True
            class_lines = [f"  {i+1}: {line}"]
        elif in_class:
            class_lines.append(f"  {i+1}: {line}")
            if len(class_lines) > 50:
                in_class = False
                print('\n'.join(class_lines))
                class_lines = []

    # 4. Read atom.py's MmaAtom definition
    print("\n=== atom.py MmaAtom class ===")
    with open(base + "/cute/atom.py") as f:
        atom_src = f.read()
    atom_lines = atom_src.split('\n')
    in_class = False
    class_lines = []
    for i, line in enumerate(atom_lines):
        if 'class MmaAtom' in line:
            in_class = True
            class_lines = [f"  {i+1}: {line}"]
        elif in_class:
            class_lines.append(f"  {i+1}: {line}")
            # Stop at another top-level class
            if len(class_lines) > 1 and line.startswith('class '):
                in_class = False
                print('\n'.join(class_lines))
                class_lines = []
                break
            if len(class_lines) > 100:
                in_class = False
                break
    if class_lines:
        print('\n'.join(class_lines))

    # 5. Look in cute module for anything like "make_smem_desc_tensor" or desc patterns
    print("\n=== cute module - anything with 'smem' or 'desc' in function names ===")
    for name in dir(cute):
        if 'smem' in name.lower() or 'desc' in name.lower() or 'frag' in name.lower():
            fn = getattr(cute, name)
            print(f"  cute.{name}: {fn}")
            try:
                print(f"    sig: {inspect.signature(fn)}")
            except:
                pass

    # 6. Read cute/__init__.py to see all exports
    print("\n=== cute/__init__.py - first 100 lines ===")
    with open(base + "/cute/__init__.py") as f:
        cute_init = f.read()
    for i, line in enumerate(cute_init.split('\n')[:100]):
        print(f"  {i+1}: {line}")


@app.local_entrypoint()
def main():
    type_probe3.remote()
