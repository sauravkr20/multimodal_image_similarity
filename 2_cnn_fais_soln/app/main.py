from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import SHOE_IMAGES_FOLDER, FAISS_INDEX_PATH, CLIP_FAISS_INDEX_PATH, CLIP_FAISS_INDEX_TEXT_PATH, CHROMA_CNN_EMBEDDINGS_COLLECTION
from app.model import extract_embedding, extract_clip_embedding, extract_clip_text_embedding
from app.search import load_index, load_embedding_metadata, save_index, search, load_product_metadata

# from app.services.cnn_faiss import CNNFaissSearch
from app.services.cnn_chroma import CNNChromaSearch
from app.services.clip_faiss import CLIPFaissSearch
from app.controllers.search_controller import SearchController
from app.controllers.products_controller import ProductsController
from app.controllers.add_controller import AddController

from app.routes import search as search_routes
from app.routes import products as products_routes
from app.routes import add as add_routes
from app.db.chroma import ChromaDBClient

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.db.chroma import ChromaDBClient
from app.services.clip_chroma import CLIPChromaSearch

# Create the ChromaDB client (persistent directory is specified here)
chroma_client = ChromaDBClient(persist_directory="../data/chroma_db")



# # Load FAISS index and embedding metadata
# index = load_index(FAISS_INDEX_PATH)
# clip_index = load_index(CLIP_FAISS_INDEX_PATH)
# clip_text_index = load_index(CLIP_FAISS_INDEX_TEXT_PATH)

# Mount static files for images
app.mount("/images", StaticFiles(directory=SHOE_IMAGES_FOLDER), name="images")

# Initialize services and controllers
# cnn_faiss_service = (index, extract_embedding, search)

clip_chroma_search = CLIPChromaSearch(
    chroma_client=chroma_client,
    extract_clip_embedding=extract_clip_embedding,
    extract_clip_text_embedding=extract_clip_text_embedding
)

cnn_chroma_search = CNNChromaSearch(
    chroma_client=chroma_client,
    collection_name=CHROMA_CNN_EMBEDDINGS_COLLECTION, 
    extract_embedding_func=extract_embedding  
)


search_controller = SearchController(cnn_chroma_search, clip_chroma_search)
products_controller = ProductsController()
# add_controller = AddController(
#     faiss_cnn_index=index, 
#     faiss_clip_index = clip_index
# )


# Inject controllers into routers
search_routes.search_controller = search_controller
products_routes.products_controller = products_controller
# add_routes.add_controller = add_controller

# Include routers
app.include_router(search_routes.router)
app.include_router(products_routes.router)
# app.include_router(add_routes.router)
