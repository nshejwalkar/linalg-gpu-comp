"""
stage1_type_probe8.py — find make_copy_atom + how to create CopyAtom from Ld32x32bOp.

Error: 'Ld32x32bOp' object has no attribute '_trait'
tc.make_tmem_copy needs a CopyAtom (has ._trait), not raw Ld32x32bOp.

Also confirm: _cute_ir.gemm passed (MMA part worked with direct call).

Find: cute.make_copy_atom / CopyAtom constructor / how Ld32x32bOp -> CopyAtom.
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-type-probe8")


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def type_probe8():
    import inspect
    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.nvgpu.tcgen05 as tc

    base = "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages/cutlass"

    print("=" * 72)
    print("TYPE PROBE 8 — CopyAtom, make_copy_atom, Ld32x32bOp._make_trait")
    print("=" * 72)

    # 1. Check cute module for copy-related APIs
    print("\n=== cute module copy-related attrs ===")
    for name in dir(cute):
        if 'copy' in name.lower() or 'atom' in name.lower():
            fn = getattr(cute, name)
            print(f"  cute.{name}: {type(fn).__name__}")
            try:
                print(f"    sig: {inspect.signature(fn)}")
            except:
                pass

    # 2. Read copy.py from tcgen05 to find Ld32x32bOp definition
    print("\n=== tcgen05/copy.py - Ld32x32bOp definition ===")
    with open(base + "/cute/nvgpu/tcgen05/copy.py") as f:
        copy_src = f.read()
    copy_lines = copy_src.split('\n')
    # Find Ld32x32bOp
    in_class = False
    class_lines = []
    for i, line in enumerate(copy_lines):
        if 'class Ld32x32bOp' in line or 'class Ld16x' in line:
            in_class = True
            class_lines = [f"  {i+1}: {line}"]
        elif in_class:
            class_lines.append(f"  {i+1}: {line}")
            if len(class_lines) > 60:
                in_class = False
                print('\n'.join(class_lines))
                class_lines = []
                if len(copy_lines) - i > 50:  # continue for next class
                    continue
                break
    if class_lines:
        print('\n'.join(class_lines))

    # 3. Read atom.py - make_copy_atom
    print("\n=== atom.py - make_copy_atom, CopyAtom ===")
    with open(base + "/cute/atom.py") as f:
        atom_src = f.read()
    atom_lines = atom_src.split('\n')
    # Find make_copy_atom
    for i, line in enumerate(atom_lines):
        if 'def make_copy_atom' in line or 'class CopyAtom' in line:
            start = i
            end = min(len(atom_lines), i + 80)
            for j, l in enumerate(atom_lines[start:end], start=start+1):
                print(f"  {j}: {l}")
            print()


@app.local_entrypoint()
def main():
    type_probe8.remote()
