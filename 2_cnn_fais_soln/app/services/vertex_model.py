from google.cloud import aiplatform

class VertexAIService:
    def __init__(self, project_id: str, location: str = "us-central1"):
        aiplatform.init(project=project_id, location=location)
        
    def get_text_embedding(self, text: str) -> np.ndarray:
        model = aiplatform.language_models.TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        embeddings = model.get_embeddings([text])
        return np.array(embeddings[0].values).astype("float32")

    def get_image_embedding(self, image_bytes: bytes) -> np.ndarray:
        model = aiplatform.vision_models.ImageEmbeddingModel.from_pretrained("imagetext@001")
        image = aiplatform.vision_models.Image(image_bytes=image_bytes)
        embedding = model.get_embeddings(image=image)
        return np.array(embedding.vector).astype("float32")
