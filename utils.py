"""
Shared dict-tensor utilities for the Loop-TNR sandbox.

All tensors in this codebase are `dict[tuple[int,...], np.ndarray]`
keyed by module labels. These helpers operate on that representation.
"""
import numpy as np


def dict_diff(d1, d2):
    """Element-wise difference of two dict-tensors with matching keys."""
    if d1.keys() != d2.keys():
        return None
    return {k: d1[k] - d2[k] for k in d1}


def dict_squared_norm(d):
    """0.5 * sum_k ||d[k]||_F^2  (half the squared Frobenius norm)."""
    return 0.5 * sum(np.linalg.norm(v) ** 2 for v in d.values())


def dict_frobenius_norm(d):
    """sqrt(sum_k ||d[k]||_F^2) — true Frobenius norm of the dict
    treated as one concatenated tensor."""
    return np.sqrt(sum(np.linalg.norm(v) ** 2 for v in d.values()))


def dict_to_vec(d, sorted_keys):
    """Flatten a dict-tensor to a 1D real vector (Re/Im interleaved per key)."""
    parts = []
    for k in sorted_keys:
        parts.append(d[k].real.ravel())
        parts.append(d[k].imag.ravel())
    return np.concatenate(parts)


def vec_to_dict(vec, sorted_keys, shapes):
    """Reconstruct a dict-tensor from a flat real vector."""
    d = {}
    i = 0
    for k, shape in zip(sorted_keys, shapes):
        size = int(np.prod(shape))
        re = vec[i:i + size].reshape(shape)
        im = vec[i + size:i + 2 * size].reshape(shape)
        d[k] = re + 1j * im
        i += 2 * size
    return d
