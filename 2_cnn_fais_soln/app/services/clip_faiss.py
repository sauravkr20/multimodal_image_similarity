from PIL import Image
from typing import List, Optional
from app.models.search_models import SearchResultItem
from app.db.mongo import embedding_clip_faiss_metadata_col, embedding_clip_faiss_text_metadata_col
import numpy as np
from app.search import get_embeddings_by_indices

class CLIPFaissSearch:
    def __init__(self, index, text_index, extract_clip_embedding, extract_clip_text_embedding, search_func):
        self.index = index
        self.text_index = text_index
        self.extract_embedding = extract_clip_embedding
        self.extract_text_embedding = extract_clip_text_embedding
        self.search = search_func

    async def search_image(self, image: Image.Image, top_k: int) -> List[SearchResultItem]:
        # Extract embedding (assumed synchronous)
        emb = self.extract_embedding(image)  # numpy array shape (dim,)
        emb = emb.reshape(1, -1).astype('float32')  # FAISS expects 2D array

        print("inside the search image clip method")

        # Perform FAISS search: FAISS returns (scores, indices)
        indices, scores = self.search(self.index, emb, top_k)
        print("the scores and indices", scores, indices)

        results: List[SearchResultItem] = []

        # Iterate over the first query's results (assuming single query)
        for idx, score in zip(indices, scores):
            # Skip invalid indices
            if idx == -1:
                continue

            embedding_doc = embedding_clip_faiss_metadata_col.find_one({"faiss_index": int(idx)})
            if not embedding_doc:
                continue

            image_id = embedding_doc.get("image_id")
            image_path = embedding_doc.get("image_path")
            item_id = embedding_doc.get("item_id")

            results.append({
                "image_id": str(image_id),
                "item_id": item_id,
                "image_path": image_path,
                "score": float(score)
            })

        return results
    
    async def search_image_text(
        self,
        image: Image.Image,
        query_text: Optional[str],
        top_k: int,
        text_weight: float = 0.4,
        image_weight: float = 0.6,
    ) -> List[SearchResultItem]:

        # Extract image embedding
        image_emb = self.extract_embedding(image).reshape(1, -1).astype("float32")

        # Search image embedding against image index
        indices, distances = self.search(self.index, image_emb, 1000)
        valid_indices = [idx for idx in indices if idx != -1]

        if not query_text:
            return self._build_results(valid_indices, distances)

        # Extract text embedding for query
        query_text_emb = self.extract_text_embedding(query_text)

        # Fetch image metadata for valid indices
        metadata_docs = list(
            embedding_clip_faiss_metadata_col.find({"faiss_index": {"$in": valid_indices}})
        )
        faiss_index_to_meta = {doc["faiss_index"]: doc for doc in metadata_docs}

        # Get item_ids from image metadata
        item_ids = [doc["item_id"] for doc in metadata_docs]

        # Fetch corresponding text metadata for item_ids
        text_docs = list(
            embedding_clip_faiss_text_metadata_col.find({"item_id": {"$in": item_ids}})
        )
        item_id_to_text_faiss_index = {
            doc["item_id"]: doc["faiss_index"] for doc in text_docs if "faiss_index" in doc
        }

        # Batch load text embeddings
        text_faiss_indices = list(item_id_to_text_faiss_index.values())
        text_embeddings_batch = get_embeddings_by_indices(self.text_index, text_faiss_indices)

        # Map text faiss_index -> embedding
        text_index_to_emb = {
            idx: emb for idx, emb in zip(text_faiss_indices, text_embeddings_batch)
        }

        # Compute combined scores
        combined_results = []

        for idx, dist in zip(indices, distances):
            if idx == -1:
                continue

            meta_doc = faiss_index_to_meta.get(idx)
            if not meta_doc:
                continue

            item_id = meta_doc["item_id"]
            image_id = meta_doc["image_id"]
            image_path = meta_doc["image_path"]

            text_faiss_idx = item_id_to_text_faiss_index.get(item_id)
            text_emb = text_index_to_emb.get(text_faiss_idx)

            text_sim = self._cosine_similarity(query_text_emb, text_emb) if text_emb is not None else 0.0
            img_sim = 1.0 - dist  # distance to similarity

            combined_score = image_weight * img_sim + text_weight * text_sim

            combined_results.append({
                "image_id": str(image_id),
                "item_id": item_id,
                "image_path": image_path,
                "image_score": float(img_sim),
                "text_score": float(text_sim),
                "combined_score": float(combined_score),
            })

        # Sort by combined score
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)

        return combined_results[:top_k]