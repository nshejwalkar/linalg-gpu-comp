"""
stage1_type_probe.py — probe what make_umma_smem_desc returns INSIDE a kernel.

The MLIR error from v6 was:
  'cute.gemm' op expects memref/smem_desc_view for operand A,
   but gets A:!cute.memref<bf16, smem...>

So MLIR cute.gemm DOES accept smem_desc_view. The blocker is Python-level:
_normalize_variadic_tensor_operand checks isinstance(x, Tensor).

We need to find: does make_umma_smem_desc return a TensorSSA (subclass of Tensor)
inside a @cute.kernel trace? Or do we need another wrapper?

Also probes:
1. type(desc_val) inside kernel
2. cute module attrs containing "desc" or "Tensor"
3. _cute_nvgpu_ir direct call
4. mma_atom_call signature
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-type-probe")


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def type_probe():
    """Probe return type of make_umma_smem_desc inside kernel + other patterns."""
    import inspect
    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.nvgpu.tcgen05 as tc
    from cutlass.cutlass_dsl import BFloat16, Float32, Uint32
    from cutlass.cute.nvgpu.common import OperandMajorMode

    print("=" * 72)
    print("TYPE PROBE — make_umma_smem_desc inside kernel")
    print("=" * 72)

    # 1. Check cute module for relevant attrs
    print("\n=== cute module attrs with 'desc' or 'Tensor' or 'ssa' ===")
    for a in dir(cute):
        if any(k in a.lower() for k in ['desc', 'tensor', 'ssa', 'smem']):
            print(f"  cute.{a}")

    # 2. Check _cute_ir module (the MLIR binding)
    print("\n=== _cute_ir module ===")
    try:
        import cutlass._mlir._mlir_libs._cutlass_ir._mlir.ir as mlir_ir
        print(f"  ir.Value type: {mlir_ir.Value}")
    except Exception as e:
        print(f"  ir import: {e}")

    # 3. Check cute.Tensor and its subclasses
    print("\n=== cute.Tensor class hierarchy ===")
    print(f"  cute.Tensor: {cute.Tensor}")
    print(f"  cute.Tensor MRO: {[c.__name__ for c in cute.Tensor.__mro__]}")
    # Check if TensorSSA exists
    for name in ['TensorSSA', 'TensorValue', 'SmemDescTensor', 'DescTensor']:
        val = getattr(cute, name, None)
        if val:
            print(f"  cute.{name}: {val}")

    # 4. Inspect what make_umma_smem_desc signature says in detail
    print("\n=== make_umma_smem_desc full inspect ===")
    fn = getattr(tc, 'make_umma_smem_desc', None)
    if fn:
        print(f"  sig: {inspect.signature(fn)}")
        print(f"  return: {inspect.signature(fn).return_annotation}")
        try:
            src = inspect.getsource(fn)
            print(f"  source:\n{src[:1500]}")
        except Exception as e:
            print(f"  source: {e}")

    # 5. Look for mma_atom_call in cute module
    print("\n=== cute.mma_atom_call / cute.atom ===")
    mma_atom_call = getattr(cute, 'mma_atom_call', None)
    print(f"  cute.mma_atom_call: {mma_atom_call}")
    try:
        from cutlass.cute.atom import mma_atom_call as mac
        print(f"  atom.mma_atom_call sig: {inspect.signature(mac)}")
    except Exception as e:
        print(f"  atom.mma_atom_call: {e}")

    # 6. Look for lower-level _cute_nvgpu_ir
    print("\n=== _cute_nvgpu_ir module ===")
    try:
        import cutlass._mlir._mlir_libs._cutlass_ir._mlir.ir as ir
        import cutlass.cute._cute_nvgpu_ir as nvgpu_ir
        print(f"  nvgpu_ir: {dir(nvgpu_ir)[:20]}")
    except Exception as e:
        print(f"  nvgpu_ir: {e}")

    # 7. Look in tcgen05 module source for make_umma_smem_desc impl
    print("\n=== tcgen05 module source search ===")
    try:
        import cutlass.cute.nvgpu.tcgen05 as tc_mod
        import os
        tc_file = inspect.getfile(tc_mod)
        print(f"  tcgen05 file: {tc_file}")
        with open(tc_file) as f:
            content = f.read()
        # Find make_umma_smem_desc
        lines = content.split('\n')
        in_fn = False
        fn_lines = []
        for i, line in enumerate(lines):
            if 'def make_umma_smem_desc' in line:
                in_fn = True
                fn_lines = [f"  {i+1}: {line}"]
            elif in_fn:
                fn_lines.append(f"  {i+1}: {line}")
                if len(fn_lines) > 40:
                    in_fn = False
                    print('\n'.join(fn_lines))
                    break
        if fn_lines and in_fn:
            print('\n'.join(fn_lines))
    except Exception as e:
        print(f"  tcgen05 source search: {e}")

    # 8. Probe cute.make_tensor with different arg types
    print("\n=== cute.make_tensor signature ===")
    mt = getattr(cute, 'make_tensor', None)
    if mt:
        try:
            print(f"  sig: {inspect.signature(mt)}")
        except Exception as e:
            print(f"  make_tensor sig: {e}")

    # 9. Check algorithm.py (where gemm is defined) for smem_desc handling
    print("\n=== algorithm.py smem_desc section ===")
    try:
        import cutlass.cute.algorithm as alg_mod
        alg_file = inspect.getfile(alg_mod)
        with open(alg_file) as f:
            content = f.read()
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'desc' in line.lower() or 'smem_desc' in line.lower():
                print(f"  {i+1}: {line}")
    except Exception as e:
        print(f"  algorithm.py: {e}")


@app.local_entrypoint()
def main():
    type_probe.remote()
