from app.db.chroma import ChromaDBClient
import numpy as np
from typing import List, Optional
from app.db.mongo import embedding_cnn_faiss_metadata_col
from app.models.search_models import SearchResultItem
from app.search import get_embeddings_by_indices  # Adjust if needed
from app.config import CHROMA_CLIP_IMAGE_EMBEDDINGS_COLLECTION, CHROMA_CLIP_ITEM_EMBEDDINGS_COLLECTION

class CLIPChromaSearch:
    def __init__(self, chroma_client: ChromaDBClient, extract_clip_embedding, extract_clip_text_embedding):
        self.chroma_client = chroma_client
        self.extract_clip_embedding = extract_clip_embedding
        self.extract_clip_text_embedding = extract_clip_text_embedding

    async def search_image(self, image, top_k: int) -> List[object]:
        # Extract image embedding
        emb = self.extract_clip_embedding(image)
        emb = emb.reshape(1, -1).astype("float32") 
        
        results = self.chroma_client.search_embeddings(
            collection_name=CHROMA_CLIP_IMAGE_EMBEDDINGS_COLLECTION,
            query_embedding=emb[0],
            top_k=top_k
        )

        image_results = results['metadatas']
        distances = results['distances']

        results = []
        for i, metadata in enumerate(image_results):
            item_id = metadata.get("item_id")
            image_id = metadata.get("image_id")
            image_path = metadata.get("image_path")
            
            results.append({
                "image_id": str(image_id),
                "item_id": item_id,
                "image_path": image_path,
                "score": float(distances[i])
            })
        
        return results
    
    async def search_image_and_text(
        self,
        image,
        query_text: Optional[str],
        top_k: int,
        min_text_weight: float = 0.1,
        max_text_weight: float = 0.6,
    ) -> List[SearchResultItem]:
        # Extract image embedding
        image_emb = self.extract_clip_embedding(image).reshape(1, -1).astype("float32")
        
        # Search image embeddings
        image_results = self.chroma_client.search_embeddings(
            collection_name=CHROMA_CLIP_IMAGE_EMBEDDINGS_COLLECTION,
            query_embedding=image_emb[0],
            top_k=top_k * 5  # Search more to allow re-ranking after fusion
        )   

        # Extract item_ids from image results
        item_ids = list({result.get("item_id") for result in image_results['metadatas'][0]})

        # Fetch corresponding text embeddings for the items
        text_results = self.chroma_client.get_item_embeddings(
            item_ids=item_ids,
            collection_name=CHROMA_CLIP_ITEM_EMBEDDINGS_COLLECTION
        )

        # print(text_results)
        
        text_embeddings = {}
        for idx, item_id in enumerate(text_results['ids']):
            text_embeddings[item_id] = text_results['embeddings'][idx]

        # Extract query text embedding once if query_text exists
        query_text_emb = None
        if query_text and query_text.strip():
            query_text_emb = self.extract_clip_text_embedding(query_text)

        # Compute raw image and text scores lists for normalization
        raw_img_scores = []
        raw_text_scores = []
        combined_results = []

        for idx, image_metadata in enumerate(image_results['metadatas'][0]):
            item_id = image_metadata.get("item_id")
            image_id = image_metadata.get("image_id")
            image_path = image_metadata.get("image_path")
            img_score = 1.0 - image_results['distances'][0][idx]  # distance to similarity
            raw_img_scores.append(img_score)

            text_emb = text_embeddings.get(item_id)
            text_score = 0.0
            if query_text_emb is not None and text_emb is not None:
                text_score = self._cosine_similarity(query_text_emb, text_emb)
            raw_text_scores.append(text_score)

            combined_results.append({
                "image_id": str(image_id),
                "item_id": item_id,
                "image_path": image_path,
                "image_score": img_score,
                "text_score": text_score,
                "combined_score": 0.0  # placeholder
            })

        # Normalize scores using min-max normalization
        def min_max_normalize(scores):
            min_s = np.min(scores)
            max_s = np.max(scores)
            if max_s - min_s < 1e-8:
                return np.zeros_like(scores)
            return (scores - min_s) / (max_s - min_s)

        norm_img_scores = min_max_normalize(np.array(raw_img_scores))
        norm_text_scores = min_max_normalize(np.array(raw_text_scores))

        # Adaptive weighting based on query_text presence
        def get_adaptive_weights(query_text):
            if not query_text or len(query_text.strip()) < 3:
                return 1.0, 0.0 
            else:
                return 0.6, 0.4 

        image_weight, text_weight = self.get_confidence_weights(norm_img_scores, norm_text_scores, query_text, top_k=20)


        # Compute combined scores with adaptive weights
        for i, result in enumerate(combined_results):
            combined_score = image_weight * norm_img_scores[i] + text_weight * norm_text_scores[i]
            result["combined_score"] = float(combined_score)

        # Sort by combined score descending
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)

        top_k_results = combined_results[:top_k]
        top_k_image_ids = [int(result["image_id"]) for result in top_k_results]

        # Fetch metadata for top_k_image_ids (optional, depends on your setup)
        metadata_docs = list(
            embedding_cnn_faiss_metadata_col.find({"image_id": {"$in": top_k_image_ids}})
        )

        image_path_map = {
            str(doc["image_id"]): doc.get("image_path") for doc in metadata_docs
        }

        for result in top_k_results:
            # Update image_path if available from metadata
            if not result.get("image_path"):
                result["image_path"] = image_path_map.get(result["image_id"], None)

        # print(f"top_k_results: {top_k_results}")

        return top_k_results


    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def mean_top_k(self, scores: np.ndarray, k: int = 20) -> float:
        if len(scores) == 0:
            return 0.0
        sorted_scores = np.sort(scores)[::-1]  # descending order
        top_k_scores = sorted_scores[:min(k, len(sorted_scores))]
        return np.mean(top_k_scores)

    def get_confidence_weights(self, norm_img_scores, norm_text_scores, query_text, top_k=20):
        if not query_text or len(query_text.strip()) < 3:
            # No or very short query text: full weight on image
            return 1.0, 0.0
        
        img_confidence = self.mean_top_k(norm_img_scores, top_k)
        text_confidence = self.mean_top_k(norm_text_scores, top_k)
        
        total_confidence = img_confidence + text_confidence
        if total_confidence < 1e-8:
            # fallback to equal weights if both zero
            return 0.5, 0.5
        
        image_weight = img_confidence / total_confidence
        text_weight = text_confidence / total_confidence
        
        return image_weight, text_weight
