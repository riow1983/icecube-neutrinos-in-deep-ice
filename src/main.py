# %% [code] {"jupyter":{"outputs_hidden":false}}
import getpass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum
import pytorch_lightning as pl
from tta import make_predictions


KERNEL = False if getpass.getuser() == "anjum" else True
COMP_NAME = "icecube-neutrinos-in-deep-ice"

if not KERNEL:
    INPUT_PATH = Path(f"/mnt/storage_dimm2/kaggle_data/{COMP_NAME}")
    OUTPUT_PATH = Path(f"/mnt/storage_dimm2/kaggle_output/{COMP_NAME}")
    MODEL_CACHE = Path("/mnt/storage/model_cache/torch")
    TRANSPARENCY_PATH = INPUT_PATH / "ice_transparency.txt"
else:
    INPUT_PATH = Path(f"/kaggle/input/{COMP_NAME}")
    MODEL_CACHE = None
    TRANSPARENCY_PATH = "/kaggle/input/icecubetransparency/ice_transparency.txt"

    # Install packages
    import subprocess

    whls = [
        "/kaggle/input/pytorchgeometric/torch_cluster-1.6.0-cp37-cp37m-linux_x86_64.whl",
        "/kaggle/input/pytorchgeometric/torch_scatter-2.1.0-cp37-cp37m-linux_x86_64.whl",
        "/kaggle/input/pytorchgeometric/torch_sparse-0.6.16-cp37-cp37m-linux_x86_64.whl",
        "/kaggle/input/pytorchgeometric/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl",
        "/kaggle/input/pytorchgeometric/torch_geometric-2.2.0-py3-none-any.whl",
        "/kaggle/input/pytorchgeometric/ruamel.yaml-0.17.21-py3-none-any.whl",
    ]

    for w in whls:
        print("Installing", w)
        subprocess.call(["pip", "install", w, "--no-deps", "--upgrade"])

    import sys
    sys.path.append("/kaggle/input/graphnet/graphnet-main/src")



GLOBAL_POOLINGS = {
    "min": scatter_min,
    "max": scatter_max,
    "sum": scatter_sum,
    "mean": scatter_mean,
}

_dtype = {
    "batch_id": "int16",
    "event_id": "int64",
}



if __name__ == "__main__":
    pl.seed_everything(48, workers=True)

    model_folders = [
        "20230131-084311",
    ]

    if KERNEL:
        dataset_paths = [Path(f"../input/icecube-{f}") for f in model_folders]
    else:
        dataset_paths = [OUTPUT_PATH / f for f in model_folders]

    predictions = make_predictions(dataset_paths, mode="test")