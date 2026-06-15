"""
stage1_type_probe6.py — read algorithm.py gemm FULL, understand smem_desc_view dispatch.

Key question: after _normalize_variadic_tensor_operand, what does cute.gemm do?
Does it call _cute_ir.gemm directly or mma_atom_call?
Does the MLIR op accept the raw ir.Value from mma_make_fragment?

Also: look at atom.py make_fragment_A return type — it returns ir.OpResult.
How does _cute_ir.gemm get called? Does it accept ir.Value or cute.Tensor?

Also: test inside a kernel whether make_fragment_A(sA) then wrap as Tensor.
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-type-probe6")


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def type_probe6():
    import inspect
    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.nvgpu.tcgen05 as tc

    base = "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages/cutlass"

    print("=" * 72)
    print("TYPE PROBE 6 — algorithm.py gemm full + atom.py make_mma_atom")
    print("=" * 72)

    # 1. Read algorithm.py fully (it's short)
    print("\n=== algorithm.py FULL ===")
    with open(base + "/cute/algorithm.py") as f:
        alg_src = f.read()
    print(alg_src[:8000])

    # 2. Read atom.py around make_fragment_A return type + make_mma_atom
    print("\n=== atom.py lines 395-470 (make_fragment_A + make_fragment_B + make_fragment_C) ===")
    with open(base + "/cute/atom.py") as f:
        atom_src = f.read()
    atom_lines = atom_src.split('\n')
    for i, line in enumerate(atom_lines[394:480], start=395):
        print(f"  {i}: {line}")

    # 3. Check cute.Tensor class
    print("\n=== cute.Tensor (typing.Tensor) class ===")
    with open(base + "/cute/typing.py") as f:
        typing_src = f.read()
    # Find Tensor class
    lines = typing_src.split('\n')
    in_class = False
    class_lines = []
    for i, line in enumerate(lines):
        if 'class Tensor' in line and 'ABC' in line:
            in_class = True
            class_lines = [f"  {i+1}: {line}"]
        elif in_class:
            class_lines.append(f"  {i+1}: {line}")
            if len(class_lines) > 1 and line.startswith('class ') and 'Tensor' not in line:
                in_class = False
                print('\n'.join(class_lines[:-1]))
                break
            if len(class_lines) > 100:
                in_class = False
                break
    if class_lines:
        print('\n'.join(class_lines))


@app.local_entrypoint()
def main():
    type_probe6.remote()
