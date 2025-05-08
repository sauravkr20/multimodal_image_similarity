from app.startup import build_cnn_faiss_index, build_products_col, build_clip_faiss_index, build_clip_text_faiss_index
from app.startup_with_chroma import build_clip_item_collection, build_cnn_image_collection, build_clip_image_collection
from app.db.chroma import ChromaDBClient


chroma_client = ChromaDBClient(persist_directory="../data/chroma_db")


if __name__ == "__main__":
    build_products_col()

    # build_cnn_faiss_index()
    # build_clip_text_faiss_index()
    # build_clip_faiss_index()

    # build_cnn_image_collection(chroma_client)
    # build_clip_item_collection(chroma_client)
    build_clip_image_collection(chroma_client)