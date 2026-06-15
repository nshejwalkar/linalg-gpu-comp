"""Probe 4: dump RAW enum blocks (no parsing) for the cublasLt + data-type enums."""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
)
app = modal.App("cublaslt-probe4", image=image)


@app.function(gpu="B200", timeout=600)
def probe():
    import glob

    def raw(path, marker, before=0, after=2000):
        try:
            with open(path) as f:
                txt = f.read()
        except Exception as e:
            print(f"  cannot read {path}: {e}")
            return
        i = txt.find(marker)
        if i < 0:
            print(f"  [{marker}] not found in {path}")
            return
        # find enclosing enum block: from 'typedef enum' before, to matching name after
        start = txt.rfind("typedef enum", 0, i)
        if start < 0:
            start = txt.rfind("enum", 0, i)
        b0 = txt.find("{", start)
        b1 = txt.find("}", b0)
        print(f"\n===== {marker}  ({path}) =====")
        print(txt[b0:b1 + 60])

    inc = "/usr/local/lib/python3.11/site-packages/nvidia/cu13/include"
    raw(f"{inc}/cublasLt.h", "cublasLtMatmulDescAttributes_t")
    raw(f"{inc}/cublasLt.h", "cublasLtMatrixLayoutAttribute_t")
    raw(f"{inc}/cublasLt.h", "cublasLtMatmulPreferenceAttributes_t")
    raw(f"{inc}/cublasLt.h", "cublasLtOrder_t")
    libt = glob.glob("/usr/local/lib/python3.11/site-packages/nvidia/*/include/library_types.h")
    if libt:
        raw(libt[0], "cudaDataType_t")


@app.local_entrypoint()
def main():
    probe.remote()
