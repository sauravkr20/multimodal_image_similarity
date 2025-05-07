import faiss
import json
import numpy as np
from typing import List, Tuple

def load_index(index_path: str) -> faiss.IndexFlatIP:
    return faiss.read_index(index_path)

def save_index(index: faiss.IndexFlatIP, index_path: str):
    faiss.write_index(index, index_path)

def load_image_paths(json_path: str) -> List[str]:
    with open(json_path, "r") as f:
        return json.load(f)

def load_embedding_metadata(json_path: str) -> List[str]:
    with open(json_path, "r") as f:
        return json.load(f)
        
def load_product_metadata(json_path: str)-> List[dict]:
    with open(json_path, "r") as f:
        return json.load(f)

def save_image_paths(paths: List[str], json_path: str):
    with open(json_path, "w") as f:
        json.dump(paths, f, indent=2)

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def search(index: faiss.IndexFlatIP, query_emb: np.ndarray, top_k: int = 5) -> Tuple[List[int], List[float]]:
    D, I = index.search(query_emb.reshape(1, -1), top_k)
    return I[0].tolist(), D[0].tolist()

def get_embeddings_by_indices(index: faiss.IndexFlatIP, indices: List[int]) -> np.ndarray:
    """
    Retrieves original embeddings from FAISS index for the given indices.
    """
    if not hasattr(index, 'xb'):
        raise ValueError("This FAISS index does not store embeddings directly.")

    total = index.ntotal
    valid_indices = [i for i in indices if 0 <= i < total]

    if len(valid_indices) != len(indices):
        print("Warning: Some indices were out of bounds and have been ignored.")

    return np.array([index.xb[i] for i in valid_indices])
