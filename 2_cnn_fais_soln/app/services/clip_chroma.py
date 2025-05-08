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
        text_weight: float = 0.4,
        image_weight: float = 0.6,
    ) -> List[SearchResultItem]:
        # Extract image embedding
        image_emb = self.extract_clip_embedding(image).reshape(1, -1).astype("float32")
        
        # Search image embeddings
        image_results = self.chroma_client.search_embeddings(
            collection_name=CHROMA_CLIP_IMAGE_EMBEDDINGS_COLLECTION,
            query_embedding=image_emb[0],
            top_k=top_k
        )   

        # print(f"image results {image_results}")
        
        # Extract item_ids from image results
        item_ids = list({result.get("item_id") for result in image_results['metadatas'][0]})


        # Fetch corresponding text embeddings for the items
        text_results = self.chroma_client.get_item_embeddings(
            item_ids=item_ids,
            collection_name=CHROMA_CLIP_ITEM_EMBEDDINGS_COLLECTION
        )
        print(" ")
        print(f"text results: {text_results}")
        print(" ")
        text_embeddings = {}
        for idx, item_id in enumerate(text_results['ids']):
            text_embeddings[item_id] = text_results['embeddings'][idx]


        combined_results = []
        for idx, image_metadata in enumerate(image_results['metadatas'][0]):
            # print(f"image metadata: {image_metadata}")
            item_id = image_metadata.get("item_id")
            image_id = image_metadata.get("image_id")
            image_path = image_metadata.get("image_path")
            img_score = 1.0 - image_results['distances'][0][idx]  # convert distance to similarity
            
            # Get the corresponding text embedding
            text_emb = text_embeddings.get(item_id)
            text_score = 0.0
            if text_emb is not None:
                text_score = self._cosine_similarity(self.extract_clip_text_embedding(query_text), text_emb)
            
            # Compute the weighted sum of scores
            combined_score = (image_weight * img_score) + (text_weight * text_score)
            
            combined_results.append({
                "image_id": str(image_id),
                "item_id": item_id,
                "image_path": image_path,
                "image_score": float(img_score),
                "text_score": float(text_score),
                "combined_score": float(combined_score)
            })

        # Sort by combined score
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)

        top_k_results = combined_results[:top_k]
        top_k_image_ids = [int(result["image_id"]) for result in top_k_results]
        
        print(f"top_k_image_ids: {top_k_image_ids}")

        # Fetch metadata for top_k_image_ids
        metadata_docs = list(
            embedding_cnn_faiss_metadata_col.find({"image_id": {"$in": top_k_image_ids}})
        )

        image_path_map = {
            str(doc["image_id"]): doc.get("image_path") for doc in metadata_docs
        }


        for result in top_k_results:
            result["image_path"] = image_path_map.get(result["image_id"], None)

        print(f"top_k_results: {top_k_results}")

        return top_k_results
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
