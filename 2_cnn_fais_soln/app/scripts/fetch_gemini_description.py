
from app.db.mongo import products_col, embedding_cnn_faiss_metadata_col
import asyncio
import os 
from app.services.gemini_description import GeminiDescriptionService    
from typing import Dict, List
from app.config import SHOE_IMAGES_FOLDER


BATCH_SIZE = 100

total_products = products_col.count_documents({})  
def fetch_products_batch(skip: int, limit: int) -> List[dict] : 
    """
    Fetch a batch of products from MongoDB
    """
    cursor = products_col.find({}, skip=skip, limit=limit)
    products = []
    for doc in cursor: 
        products.append(doc)
    return products


async def read_image_bytes(image_path: str) -> bytes:
    """
    Read image bytes from a local file path. """
    from pathlib import Path
    path = Path(os.path.join(SHOE_IMAGES_FOLDER, image_path))
    if path.exists():
        # print(f"Image found: {image_path}")
        return path.read_bytes()
    else:
        print(f"Image not found: {image_path}")
        return b""
    

def fetch_image_paths(image_ids: List[int]) -> Dict[int, str]:
    """
    Given a list of image_ids, fetch their corresponding image_path from embedding_cnn_faiss_metadata_col.
    Returns a dict mapping image_id -> image_path.
    """
    cursor = embedding_cnn_faiss_metadata_col.find({"image_id": {"$in": image_ids}})
    image_paths = {}
    for doc in cursor:
        image_paths[doc["image_id"]] = doc["image_path"]
    return image_paths


async def process_products_and_update_descriptions(gemini_service: GeminiDescriptionService):
    skip = 0
    processed_count = 0
    # single test 
    # BATCH_SIZE = 1
    while (True):
    # while skip == 0:
        products = fetch_products_batch(skip = skip, limit= BATCH_SIZE)

        if not products: 
            print("No more products to process")
            break

        for product in products:
            product_id = product.get("_id")
            image_ids = product.get("other_image_id", [])
            image_ids.append(product.get("main_image_id")) if product.get("main_image_id") else None

            if not image_ids:
                print(f"Product with {product_id} has no other images")

            image_paths_map = fetch_image_paths(image_ids)
            valid_image_paths = [image_paths_map[iid] for iid in image_ids if iid in image_paths_map]


            images_bytes_list = await asyncio.gather(*[read_image_bytes(img) for img in valid_image_paths])

            images_bytes_list = [b for b in images_bytes_list if b]
            description = await gemini_service.generate_description(image_bytes_list=images_bytes_list)

            if description: 
                update_result = products_col.update_one(
                    {"_id": product_id},
                    {"$set": {"description": description}}
                )

                if update_result.modified_count == 1: 
                    print(f"Updated description for product {product_id} with gemini description: {description}")

                else: 
                    print(f"Failed to update description for product {product_id}")

            else: 
                print(f"Failed to generate gemini description for product {product_id}")

            processed_count += 1
            print(f"{processed_count} / {total_products} items done")


        skip += BATCH_SIZE