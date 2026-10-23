"""Distributed capture hub.

Two capture laptops each run ``node_agent.py`` against one camera, process
locally, and POST the resulting embeddings here. This package pairs them in
time, fuses the two viewing angles, matches against the local gallery, and
serves a browser dashboard over Server-Sent Events.

Only embeddings and small thumbnails cross the network - never raw video.
"""

from .protocol import (
    HUB_DEFAULT_PORT,
    PROTOCOL_VERSION,
    decode_vector,
    encode_vector,
)

__all__ = [
    "HUB_DEFAULT_PORT",
    "PROTOCOL_VERSION",
    "decode_vector",
    "encode_vector",
]
