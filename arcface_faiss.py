"""Compatibility shim - embedding and matching now live in
pipeline/stage4_embed.py and pipeline/stage5_match.py.

Use ``enroll.py`` for day-to-day database management.
"""

from pipeline.stage4_embed import (  # noqa: F401
    ArcFaceEmbedder,
    extract_embedding,
    get_embedder,
    l2_normalize,
)
from pipeline.stage5_match import (  # noqa: F401
    Match,
    ThreatIndex,
    get_index,
    query_cctv_crop,
    register_known_subject,
)

__all__ = [
    "ArcFaceEmbedder",
    "extract_embedding",
    "get_embedder",
    "l2_normalize",
    "Match",
    "ThreatIndex",
    "get_index",
    "query_cctv_crop",
    "register_known_subject",
]


if __name__ == "__main__":
    print(get_index().stats())
    print("identities:", get_index().identities or "(none enrolled)")
    print("\nManage the database with:  python enroll.py --help")
