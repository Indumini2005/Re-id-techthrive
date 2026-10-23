"""Pure-numpy vector helpers.

Separated from stage4_embed so that code which only *handles* embeddings -
the hub server, the fusion logic, the vector DB - does not have to import
OpenCV or ONNX Runtime. The hub laptop needs numpy and nothing else.
"""

import numpy as np


def l2_normalize(vectors, axis=-1, eps=1e-10):
    """Scale to unit length so a dot product is cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norm = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return vectors / np.maximum(norm, eps)


def cosine(a, b):
    """Cosine similarity between two vectors of any scaling."""
    a = l2_normalize(np.asarray(a, dtype=np.float32).reshape(-1))
    b = l2_normalize(np.asarray(b, dtype=np.float32).reshape(-1))
    return float(np.dot(a, b))
