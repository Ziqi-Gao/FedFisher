"""Foreground runtime setup inside the centrally supplied CPU allocation."""
import os
from pathlib import Path

from .catalog import PROJECT_ROOT, SCRATCH_ROOT
from .errors import AdapterError


def configure_runtime(job_id, cpu_cores):
    source_root = Path(__file__).resolve().parents[2]
    if source_root != PROJECT_ROOT.resolve(strict=True):
        raise AdapterError("adapter must execute from the registered FedFisher code root")
    runtime = SCRATCH_ROOT / "effect_modifier_experiment" / "server_scheduler" / job_id
    root = SCRATCH_ROOT.resolve(strict=True)
    if runtime.resolve(strict=False) != runtime or root not in runtime.parents:
        raise AdapterError("unsafe ServerScheduler runtime path")
    current = runtime
    while current != root:
        if current.exists() and current.is_symlink():
            raise AdapterError("runtime path may not contain symlinks")
        current = current.parent
    runtime.mkdir(parents=True, exist_ok=True)
    for path in (runtime,):
        if path.is_symlink() or path.resolve(strict=True) != path:
            raise AdapterError("runtime path may not resolve through symlinks")
    for name in ("tmp", "python_cache", "xdg_cache", "torch_cache"):
        child = runtime / name
        child.mkdir(exist_ok=True)
        if child.is_symlink() or child.resolve(strict=True) != child:
            raise AdapterError("runtime child may not resolve through symlinks")
    os.environ.update(
        {
            "OMP_NUM_THREADS": str(cpu_cores),
            "MKL_NUM_THREADS": str(cpu_cores),
            "OPENBLAS_NUM_THREADS": str(cpu_cores),
            "NUMEXPR_NUM_THREADS": str(cpu_cores),
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "TMPDIR": str(runtime / "tmp"),
            "PYTHONPYCACHEPREFIX": str(runtime / "python_cache"),
            "XDG_CACHE_HOME": str(runtime / "xdg_cache"),
            "TORCH_HOME": str(runtime / "torch_cache"),
        }
    )
