
from app.db.chroma import ChromaDBClient
from app.config import gemini_api_key
from app.services.gemini_description import GeminiDescriptionService
import asyncio
from app.scripts.fetch_gemini_description import process_products_and_update_descriptions
from app.startup_with_chroma import build_clip_item_collection


chroma_client = ChromaDBClient(persist_directory="../data/chroma_db")


if __name__ == "__main__":
    # build_products_col()

    # build_cnn_faiss_index()
    # build_clip_text_faiss_index()
    # build_clip_faiss_index()

    # build_cnn_image_collection(chroma_client)
    build_clip_item_collection(chroma_client)
    # build_clip_image_collection(chroma_client)

    # gemini_service = GeminiDescriptionService(gemini_api_key)
    # asyncio.run(process_products_and_update_descriptions(gemini_service))