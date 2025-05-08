from PIL import Image
from typing import List
from app.models.search_models import SearchResultItem
from app.db.chroma import ChromaDBClient

class CNNChromaSearch:
    def __init__(self, chroma_client: ChromaDBClient, collection_name: str, extract_embedding_func):
        self.chroma = chroma_client
        self.collection_name = collection_name
        self.extract_embedding = extract_embedding_func

    async def search_image(self, image: Image.Image, top_k: int) -> List[SearchResultItem]:
        # Extract the embedding for the image
        emb = self.extract_embedding(image).reshape(1, -1).astype('float32')

        # Search the Chroma collection for the most similar embeddings
        results = self.chroma.search_embeddings(
            collection_name=self.collection_name,
            query_embedding=emb[0],
            top_k=top_k
        )

        search_results: List[SearchResultItem] = []

        # Extract results from ChromaDB
        for idx, score in zip(results['metadatas'], results['distances']):
            image_metadata = idx  # This contains the metadata associated with the image
            item_id = image_metadata.get("item_id")
            image_id = image_metadata.get("image_id")
            image_path = image_metadata.get("image_path")

            search_results.append({
                "image_id": str(image_id),
                "item_id": item_id,
                "image_path": image_path,
                "score": float(score)  # Ensure score is float
            })

        return search_results
