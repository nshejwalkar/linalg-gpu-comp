"""
stage1_type_probe7.py — find concrete cute.Tensor subclasses, wrapping ir.Value.

We know:
- cute.gemm calls t.value for t in a_list (from algorithm.py)
- t must be isinstance(t, cute.Tensor)
- The actual ir.Value from make_umma_smem_desc is an smem_desc_view
- We need a cute.Tensor subclass that wraps any ir.Value

Key: atom.py uses `isinstance(input, _Tensor)` - what is _Tensor?
Also: cute/tensor.py has TensorSSA - what other classes?
Also: does SmemDescTensor exist or similar?

This probe reads tensor.py and typing.py to find ALL Tensor subclasses.
"""
import modal

cutlass_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-cutlass-dsl")
)

app = modal.App("qr-type-probe7")


@app.function(gpu="B200", image=cutlass_image, timeout=90)
def type_probe7():
    import inspect
    import cutlass
    import cutlass.cute as cute

    base = "/usr/local/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages/cutlass"

    print("=" * 72)
    print("TYPE PROBE 7 — tensor.py + find _Tensor + Tensor subclasses")
    print("=" * 72)

    # 1. Read tensor.py FULL
    print("\n=== tensor.py FULL ===")
    with open(base + "/cute/tensor.py") as f:
        tensor_src = f.read()
    # Print first 200 lines
    for i, line in enumerate(tensor_src.split('\n')[:200]):
        print(f"  {i+1}: {line}")

    # 2. Find _Tensor in atom.py
    print("\n=== atom.py imports + _Tensor usage ===")
    with open(base + "/cute/atom.py") as f:
        atom_src = f.read()
    atom_lines = atom_src.split('\n')
    for i, line in enumerate(atom_lines[:30]):
        print(f"  {i+1}: {line}")
    print("  ...")
    for i, line in enumerate(atom_lines):
        if '_Tensor' in line or 'from .tensor' in line or 'from .typing' in line:
            print(f"  {i+1}: {line}")

    # 3. Read typing.py class structure
    print("\n=== typing.py - Tensor subclasses ===")
    with open(base + "/cute/typing.py") as f:
        typing_src = f.read()
    typing_lines = typing_src.split('\n')
    for i, line in enumerate(typing_lines):
        if 'class ' in line and ('Tensor' in line or 'Desc' in line):
            print(f"  {i+1}: {line}")

    # 4. Find all classes in tensor.py
    print("\n=== tensor.py - all class definitions ===")
    tensor_lines = tensor_src.split('\n')
    for i, line in enumerate(tensor_lines):
        if line.startswith('class '):
            print(f"  {i+1}: {line}")

    # 5. Try: cute.make_tensor with smem_desc ir.Value
    # We need to know if make_tensor creates a Tensor subclass
    print("\n=== cute.make_tensor source ===")
    try:
        import cutlass.cute.core as core_mod
        fn = getattr(core_mod, 'make_tensor', None)
        if fn:
            src = inspect.getsource(fn)
            print(src[:2000])
    except Exception as e:
        print(f"  {e}")

    # 6. Which module has make_tensor?
    print("\n=== Where is cute.make_tensor defined? ===")
    fn = getattr(cute, 'make_tensor', None)
    if fn:
        print(f"  module: {fn.__module__}")
        print(f"  file: {inspect.getfile(fn)}")


@app.local_entrypoint()
def main():
    type_probe7.remote()
