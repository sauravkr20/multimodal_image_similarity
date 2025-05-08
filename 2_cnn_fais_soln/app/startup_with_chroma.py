from app.model import extract_clip_embedding, extract_clip_text_embedding, extract_embedding
import os

from PIL import Image
import numpy as np
import json
import time
from pathlib import Path
from app.config import SHOE_IMAGES_FOLDER, IMAGE_PATHS_JSON, SHOE_PRODUCT_JSON_PATH, CHROMA_CLIP_IMAGE_EMBEDDINGS_COLLECTION, CHROMA_CLIP_ITEM_EMBEDDINGS_COLLECTION, CHROMA_CNN_EMBEDDINGS_COLLECTION
from app.db.chroma import ChromaDBClient  # Assuming ChromaDBClient is your Chroma client
from app.db.mongo import products_col

BATCH_SIZE = 1000

CLIP_ITEM_LOG_FILE_PATH="clip_item_build_time.log"
CLIP_IMAGE_LOG_FILE_PATH="clip_image_build_time.log"
CNN_LOG_FILE_PATH="cnn_build_time.log"


def flatten_metadata(metadata: dict, parent_key='', sep='.'):
    """
    Flatten nested metadata into a flat dictionary with dot-separated keys.
    """
    items = []
    for key, value in metadata.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_metadata(value, new_key, sep=sep).items())
        elif isinstance(value, list):
            for idx, sub_value in enumerate(value):
                items.extend(flatten_metadata(sub_value, f"{new_key}{sep}{idx}", sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def get_item_metadata_batch(item_ids):
    """
    Fetch metadata for all item_ids at once using MongoDB $in query.
    Returns a dictionary where keys are item_ids and values are their corresponding metadata.
    """
    item_ids_list = list(item_ids)
    items = products_col.find({"item_id": {"$in": item_ids_list}})
    item_metadata_map = {}
    for item in items:
        item_metadata_map[item['item_id']] = flatten_metadata(item.get('metadata', {}))

    return item_metadata_map

def log_batch_time(batch_start, batch_end, batch_start_time, total_time_ms, batch_times, log_file_path):
    """
    Helper function to log the processing time of a batch.
    """
    batch_end_time = time.perf_counter()
    batch_duration_ms = (batch_end_time - batch_start_time) * 1000
    total_time_ms += batch_duration_ms
    batch_times.append(batch_duration_ms)

    with open(log_file_path, "a") as log_file:
        log_file.write(f"Batch {batch_start} - {batch_end} processed in {batch_duration_ms:.2f} ms\n")
        log_file.write(f"Total time elapsed: {total_time_ms / 1000:.2f} seconds\n")
    
    return total_time_ms, batch_times

def build_cnn_image_collection(chroma_client: ChromaDBClient):
    with open(IMAGE_PATHS_JSON, "r") as f:
        original_metadata = json.load(f)

    print(f"Processing {len(original_metadata)} images for CNN image collection...")
    total_images = len(original_metadata)
    all_image_embeddings = []
    all_metadata_docs = []

    total_time_ms = 0
    batch_times = []

    for batch_start in range(0, len(original_metadata), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(original_metadata))
        batch_metadata = original_metadata[batch_start:batch_end]

        batch_start_time = time.perf_counter()

        # Step 1: Extract all unique item_ids in the current batch
        item_ids = set(record['item_id'] for record in batch_metadata)

        # Step 2: Fetch item metadata for all item_ids in the batch from MongoDB
        item_metadata_map = get_item_metadata_batch(item_ids)

        batch_embeddings = []
        batch_metadata_docs = []

        # Step 3: Process each image and include the corresponding metadata
        for idx, record in enumerate(batch_metadata, start=batch_start):
            relative_path = Path(record["image_path"])
            image_id = record["image_id"]
            item_id = record["item_id"]
            image_file_path = Path(os.path.join(SHOE_IMAGES_FOLDER, relative_path)) # Path(SHOE_IMAGES_FOLDER) / relative_path)

            if not image_file_path.exists():
                print(f"Image not found: {image_file_path}")
                continue

            try:
                image = Image.open(image_file_path).convert("RGB")
                emb = extract_embedding(image)

                # Get the item metadata for this image
                item_metadata = item_metadata_map.get(item_id, {})
                flattened_metadata = item_metadata

                batch_embeddings.append(emb)
                batch_metadata_docs.append({
                    "image_id": image_id,
                    "item_id": item_id,
                    **flattened_metadata
                })

                if (idx + 1) % 100 == 0 or (idx + 1) == total_images:
                    print(f"Processed {idx + 1}/{total_images} images")

            except Exception as e:
                print(f"Failed to process {relative_path}: {e}")

        # Step 4: Insert batch of image embeddings and metadata into Chroma
        if batch_embeddings:
            embeddings_np = np.stack(batch_embeddings).astype("float32")
            chroma_client.insert_embeddings(
                collection_name=CHROMA_CNN_EMBEDDINGS_COLLECTION,
                embeddings=embeddings_np,
                metadatas=batch_metadata_docs
            )

        total_time_ms, batch_times = log_batch_time(batch_start, batch_end, batch_start_time, total_time_ms, batch_times, CNN_LOG_FILE_PATH)

    print(f"Inserted {len(all_image_embeddings)} image embeddings into Chroma CNN image collection.")
    with open(CNN_LOG_FILE_PATH, "a") as log_file:
        log_file.write(f"Total time for CNN image collection: {total_time_ms / 1000:.2f} seconds\n")



def build_clip_item_collection(chroma_client: ChromaDBClient):
    with open(SHOE_PRODUCT_JSON_PATH, "r") as f:
        products = json.load(f)

    print(f"Processing {len(products)} items for CLIP item collection...")

    all_text_embeddings = []
    all_metadata_docs = []

    total_time_ms = 0
    batch_times = []
    total_items = len(products)

    for batch_start in range(0, len(products), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(products))
        batch_metadata = products[batch_start:batch_end]

        batch_start_time = time.perf_counter()

        # Step 1: Prepare item metadata and embed text
        item_ids = [product['item_id'] for product in batch_metadata]

        # Flatten and store metadata directly in the item entry for filtering
        batch_embeddings = []
        batch_metadata_docs = []

        
        for idx, product in enumerate(batch_metadata, start=batch_start):
            item_id = product["item_id"]
            metadata = product.get("metadata", {})

            # Flatten metadata
            flattened_metadata = flatten_metadata(metadata)

            try:
                text = metadata_to_text(metadata)
                emb = extract_clip_text_embedding(text=text)

                batch_embeddings.append(emb)
                meta_data_curr = {
                    "item_id": item_id,
                    **flattened_metadata
                }
                batch_metadata_docs.append(meta_data_curr)

                if (idx + 1) % 100 == 0 or (idx + 1) == total_items:
                    if idx <1000 : 
                        print(f"metadata looks like {meta_data_curr}")
                    print(f"Processed {idx + 1}/{total_items} images")
                
            except Exception as e:
                print(f"Failed to process item {item_id}: {e}")

        # Step 2: Insert batch of CLIP text embeddings and metadata into Chroma
        if batch_embeddings:
            embeddings_np = np.stack(batch_embeddings).astype("float32")
            item_ids = [meta["item_id"] for meta in batch_metadata_docs]

            chroma_client.insert_embeddings(
                collection_name=CHROMA_CLIP_ITEM_EMBEDDINGS_COLLECTION,
                ids = item_ids,
                embeddings=embeddings_np,
                metadatas=batch_metadata_docs
            )

        total_time_ms, batch_times = log_batch_time(batch_start, batch_end, batch_start_time, total_time_ms, batch_times, CLIP_ITEM_LOG_FILE_PATH)

    print(f"Inserted {len(all_text_embeddings)} CLIP item embeddings into Chroma item collection.")
    with open(CLIP_ITEM_LOG_FILE_PATH, "a") as log_file:
        log_file.write(f"Total time for CLIP item collection: {total_time_ms / 1000:.2f} seconds\n")


def build_clip_image_collection(chroma_client: ChromaDBClient):

    chroma_client.reset_collection(CHROMA_CLIP_IMAGE_EMBEDDINGS_COLLECTION)

    with open(IMAGE_PATHS_JSON, "r") as f:
        original_metadata = json.load(f)

    print(f"Processing {len(original_metadata)} images for CLIP image collection...")

    total_images = len(original_metadata)

    all_image_embeddings = []
    all_metadata_docs = []

    total_time_ms = 0
    batch_times = []

    for batch_start in range(0, len(original_metadata), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(original_metadata))
        batch_metadata = original_metadata[batch_start:batch_end]

        batch_start_time = time.perf_counter()

        # Step 1: Extract all unique item_ids in the current batch
        item_ids = set(record['item_id'] for record in batch_metadata)

        # Step 2: Fetch item metadata for all item_ids in the batch from MongoDB
        item_metadata_map = get_item_metadata_batch(item_ids)

        batch_embeddings = []
        batch_metadata_docs = []

        # Step 3: Process each image and include the corresponding metadata
        for idx, record in enumerate(batch_metadata, start=batch_start):
            relative_path = Path(record["image_path"])
            image_id = record["image_id"]
            item_id = record["item_id"]
            image_file_path = Path(os.path.join(SHOE_IMAGES_FOLDER, relative_path)) #Path(SHOE_IMAGES_FOLDER) / relative_path)

            if not image_file_path.exists():
                print(f"Image not found: {image_file_path}")
                continue

            try:
                image = Image.open(image_file_path).convert("RGB")
                emb = extract_clip_embedding(image)

                # Get the item metadata for this image
                item_metadata = item_metadata_map.get(item_id, {})
                flattened_metadata = item_metadata

                batch_embeddings.append(emb)

                meta_data_curr = {
                    "image_id": image_id,
                    "item_id": item_id,
                    **flattened_metadata
                }
                batch_metadata_docs.append(meta_data_curr)

                
                if (idx + 1) % 100 == 0 or (idx + 1) == total_images:
                    if idx <1000 : 
                        print(f"metadata looks like {meta_data_curr}")
                    print(f"Processed {idx + 1}/{total_images} images")

            except Exception as e:
                print(f"Failed to process {relative_path}: {e}")

        # Step 4: Insert batch of image embeddings and metadata into Chroma
        if batch_embeddings:
            embeddings_np = np.stack(batch_embeddings).astype("float32")
            image_ids = [str(meta["image_id"]) for meta in batch_metadata_docs]
            chroma_client.insert_embeddings(
                collection_name=CHROMA_CLIP_IMAGE_EMBEDDINGS_COLLECTION,
                ids=image_ids,
                embeddings=embeddings_np,
                metadatas=batch_metadata_docs
            )

        total_time_ms, batch_times = log_batch_time(batch_start, batch_end, batch_start_time, total_time_ms, batch_times, CLIP_IMAGE_LOG_FILE_PATH)

    print(f"Inserted {len(all_image_embeddings)} image embeddings into Chroma CLIP image collection.")
    with open(CLIP_IMAGE_LOG_FILE_PATH, "a") as log_file:
        log_file.write(f"Total time for CLIP image collection: {total_time_ms / 1000:.2f} seconds\n")


def metadata_to_text(metadata: dict) -> str:
    """
    Convert the metadata dictionary into a descriptive string for CLIP text embedding.
    """
    keys_to_include = [
        "category", "sub Category", " Style", " Stone", "stone Color", "stone Shape",
        "stone Setting", "motif", "adjustable", "weight", "pattern", "Bestseller"
    ]
    
    parts = []
    for key in keys_to_include:
        matching_key = next((k for k in metadata if k.strip().lower() == key.strip().lower()), None)
        if matching_key:
            value = str(metadata[matching_key]).strip()
            if value and value.lower() != "not found":
                clean_key = matching_key.strip()
                parts.append(f"{clean_key}: {value}")

    return ", ".join(parts)
