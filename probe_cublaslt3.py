"""
Probe 3: dump the exact enum constants needed to drive cublasLtMatmul via ctypes.
  - cublasLtMatmulDescAttributes_t (TRANSA/TRANSB/COMPUTE_TYPE/SCALE_TYPE/EPILOGUE...)
  - cublasLtMatrixLayoutAttribute_t (TYPE/ORDER/ROWS/COLS/LD/BATCH_COUNT/STRIDED_BATCH_OFFSET)
  - cublasLtMatmulPreferenceAttributes_t (MAX_WORKSPACE_BYTES)
  - cublasOperation_t (N=0,T=1)
  - cudaDataType (R_32F=0, R_16BF=14...)
  - cublasLtOrder_t (COL=0, ROW=1)
Run: conda activate modal && PYTHONUTF8=1 modal run probe_cublaslt3.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
)
app = modal.App("cublaslt-probe3", image=image)


@app.function(gpu="B200", timeout=600)
def probe():
    import glob, re

    hdr = "/usr/local/lib/python3.11/site-packages/nvidia/cu13/include/cublasLt.h"
    api = "/usr/local/lib/python3.11/site-packages/nvidia/cu13/include/cublas_api.h"
    libt = glob.glob("/usr/local/lib/python3.11/site-packages/nvidia/*/include/library_types.h")

    def dump_enum(path, enum_name, keep=None):
        with open(path) as f:
            txt = f.read()
        i = txt.find(enum_name)
        if i < 0:
            print(f"  [{enum_name}] NOT FOUND in {path}")
            return
        # find the enum body that precedes the typedef name (typedef enum {...} NAME;)
        start = txt.rfind("enum", 0, i)
        # the '{' after start
        b0 = txt.find("{", start)
        b1 = txt.find("}", b0)
        block = txt[b0 + 1:b1]
        print(f"\n  === {enum_name} ===")
        # parse "NAME = value" entries, tracking implicit increment
        cur = -1
        for raw in block.split(","):
            line = raw.strip()
            if not line:
                continue
            # strip comments
            code = re.split(r"//|/\*", line)[0].strip()
            if not code:
                continue
            m = re.match(r"([A-Z0-9_]+)\s*(=\s*(.+))?$", code)
            if not m:
                continue
            name = m.group(1)
            if m.group(3):
                valexpr = m.group(3).strip()
                try:
                    cur = int(valexpr, 0)
                except ValueError:
                    cur = valexpr  # symbolic
            else:
                if isinstance(cur, int):
                    cur += 1
            if keep is None or any(k in name for k in keep):
                print(f"    {name} = {cur}")

    print("cublasLt.h enums")
    dump_enum(hdr, "cublasLtMatmulDescAttributes_t",
              keep=["TRANSA", "TRANSB", "COMPUTE", "SCALE_TYPE", "EPILOGUE",
                    "POINTER_MODE", "EMULATION"])
    dump_enum(hdr, "cublasLtMatrixLayoutAttribute_t",
              keep=["TYPE", "ORDER", "ROWS", "COLS", "LD", "BATCH", "STRIDED"])
    dump_enum(hdr, "cublasLtMatmulPreferenceAttributes_t",
              keep=["MAX_WORKSPACE"])
    dump_enum(hdr, "cublasLtOrder_t")
    dump_enum(hdr, "cublasLtEpilogue_t", keep=["DEFAULT"])

    print("\ncublas_api.h enums")
    dump_enum(api, "cublasOperation_t")
    dump_enum(api, "cublasComputeType_t", keep=["32F", "EMULATED"])

    if libt:
        print("\nlibrary_types.h (cudaDataType)")
        dump_enum(libt[0], "cudaDataType_t", keep=["R_32F", "R_16BF", "R_16F", "R_64F"])


@app.local_entrypoint()
def main():
    probe.remote()
