import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import faiss

# 1. Initialize ArcFace via InsightFace
# Using buffalo_s or buffalo_l; 'buffalo_s' is lightweight and fast for edge devices
app = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. Initialize FAISS Index for Cosine Similarity
# ArcFace embeddings have 512 dimensions.
# Inner Product (METRIC_INNER_PRODUCT) on L2-normalized vectors is mathematically identical to Cosine Similarity.
EMBEDDING_DIM = 512
index = faiss.IndexFlatIP(EMBEDDING_DIM)

# Metadata lookup table to map FAISS internal IDs to subject tags
id_to_metadata = {}


def extract_arcface_embedding(face_bgr: np.ndarray) -> np.ndarray:
    """
    Takes a high-res face BGR image (from GFPGAN),
    extracts the 512-dimensional ArcFace embedding, and L2-normalizes it.
    """
    faces = app.get(face_bgr)
    if not faces:
        return None
    
    # Take the primary detected face
    embedding = faces[0].embedding
    
    # L2 normalize the vector for cosine similarity
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
        
    return embedding.astype(np.float32)


def register_known_subject(face_bgr: np.ndarray, subject_name: str) -> bool:
    """Adds a reference subject to the local FAISS gallery."""
    global index
    emb = extract_arcface_embedding(face_bgr)
    if emb is None:
        print(f"Failed to extract embedding for {subject_name}")
        return False
        
    vector = np.expand_dims(emb, axis=0)
    current_id = index.ntotal
    index.add(vector)
    id_to_metadata[current_id] = subject_name
    print(f"Registered {subject_name} with internal ID {current_id}")
    return True


def query_cctv_crop(face_bgr: np.ndarray, threshold: float = 0.65):
    """
    Compares a live reconstructed CCTV face against the FAISS gallery.
    Returns (match_found, subject_name, similarity_score).
    """
    emb = extract_arcface_embedding(face_bgr)
    if emb is None:
        return False, None, 0.0
        
    vector = np.expand_dims(emb, axis=0)
    
    # Search the top-1 closest match in FAISS
    similarities, indices = index.search(vector, k=1)
    
    best_score = similarities[0][0]
    best_idx = indices[0][0]
    
    if best_score >= threshold and best_idx in id_to_metadata:
        return True, id_to_metadata[best_idx], best_score
    
    return False, "Unknown", best_score


# --- Testing the Integration ---
if __name__ == "__main__":
    # 1. Simulate registering a reference profile in your gallery
    ref_face = cv2.imread("reference_person.png")
    if ref_face is not None:
        register_known_subject(ref_face, "Subject_A")

    # 2. Simulate querying a face output from GFPGAN
    query_face = cv2.imread("restored_gfpgan_output.png")
    if query_face is not None:
        matched, identity, score = query_cctv_crop(query_face, threshold=0.65)
        print(f"Match Found: {matched} | Identity: {identity} | Similarity Score: {score:.4f}")
