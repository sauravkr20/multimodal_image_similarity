import os
import gzip
import json

metadata_dir = '../../data/listings/metadata'  # Adjust path if needed
necklace_products = []

# Iterate over all listings JSON.gz files
for filename in sorted(os.listdir(metadata_dir)):
    if filename.endswith('.json.gz'):
        filepath = os.path.join(metadata_dir, filename)
        print(f"Processing {filename}...")
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                product = json.loads(line)
                product_types = product.get('product_type', [])
                types = []
                # product_type could be list of dicts or string
                if isinstance(product_types, list):
                    types = [pt.get('value', '').lower() for pt in product_types if isinstance(pt, dict)]
                elif isinstance(product_types, str):
                    types = [product_types.lower()]
                # Filter for necklaces
                if any('necklace' in t for t in types):
                    necklace_products.append({
                        'item_id': product.get('item_id'),
                        'product_type': types,
                        'item_name': product.get('item_name'),
                        'main_image_id': product.get('main_image_id'),
                        'other_image_id': product.get('other_image_id', [])
                    })

print(f"Found {len(necklace_products)} necklace products.")

half_length = len(necklace_products) // 2
necklace_products_half = necklace_products[:half_length]

with open('../../data/testSet1/limited2_necklace_products.json', 'w') as f:
    json.dump(necklace_products_half, f, indent=2)

print(f"Saved half ({half_length}) of the necklace products metadata to 'limited2_necklace_products.json'")
