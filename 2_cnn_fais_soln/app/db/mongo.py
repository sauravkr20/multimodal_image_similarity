from pymongo import MongoClient
from app.config import EMBEDDING_CLIP_FAISS_METADATA_COLLECTION, EMBEDDING_CNN_FAISS_METADATA_COLLECTION, PRODUCT_COLLECTION, EMBEDDING_CLIP_TEXT_FAISS_METADATA_COLLECTION

client = MongoClient("mongodb://localhost:27017")
db = client["visual_product_db"]

products_col = db[PRODUCT_COLLECTION]
embedding_cnn_faiss_metadata_col = db[EMBEDDING_CNN_FAISS_METADATA_COLLECTION]
embedding_clip_faiss_metadata_col = db[EMBEDDING_CLIP_FAISS_METADATA_COLLECTION]  
embedding_clip_faiss_text_metadata_col= db[EMBEDDING_CLIP_TEXT_FAISS_METADATA_COLLECTION]