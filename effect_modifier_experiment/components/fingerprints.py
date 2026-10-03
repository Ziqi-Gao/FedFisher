"""Exact array and model fingerprint formats shared by binary experiments."""
import hashlib
import numpy as np


def array_hash(values):
    values = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(str(values.dtype).encode())
    h.update(str(values.shape).encode())
    h.update(values.tobytes())
    return h.hexdigest()

def model_arrays(model):
    if hasattr(model, "predict_tau"):
        return {"effect_coefficients": np.asarray(model.coefficients, dtype=np.float64)}
    return {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}

def model_hash(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        h.update(key.encode() + b"\0" + array_hash(arrays[key]).encode())
    return h.hexdigest()
