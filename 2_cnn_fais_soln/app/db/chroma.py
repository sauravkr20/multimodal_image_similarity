import chromadb
from chromadb.config import Settings
from typing import List, Dict

import chromadb.errors


class ChromaDBClient:
    def __init__(self, persist_directory: str, tenant:str = 'default_tenant', database: str = 'visual_db'):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings = Settings()
        )

        # try:
        #     self.client.
        # except chromadb.errors.IDAlreadyExistsError:
        #     pass

        # try:
        #     self.client.create_database(name=database, tenant=tenant)
        # except chromadb.errors.IDAlreadyExistsError:
        #     pass

        # self.client.set_tenant(tenant)
        # self.client.set_database(database)


    def get_collection(self, collection_name: str):
        """Retrieve or create a collection by name."""
        return self.client.get_or_create_collection(collection_name)

    def search_embeddings(self, collection_name: str, query_embedding: List[float], top_k: int):
        """Search embeddings in a specific collection."""
        collection = self.get_collection(collection_name)
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"]
        )

    def get_metadata_by_ids(self, collection_name: str, ids: List[str]):
        """Fetch metadata for a list of IDs from a specific collection."""
        collection = self.get_collection(collection_name)
        return collection.get(ids=ids, include=["metadatas"])

    def get_embeddings_by_ids(self, collection_name: str, ids: List[str]):
        """Fetch embeddings by IDs from a specific collection."""
        collection = self.get_collection(collection_name)
        return collection.get(ids=ids, include=["embeddings"])

    def get_item_embeddings(self, collection_name: str, item_ids: List[str]):
        """Fetch embeddings and metadata by item_ids if item_id is used as ID."""
        collection = self.get_collection(collection_name)
        return collection.get(ids=item_ids, include=["embeddings", "metadatas"])

    def insert_embeddings(self, collection_name: str, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict]):
        """
        Insert embeddings into the specified collection.
        
        Args:
            collection_name (str): Name of the ChromaDB collection.
            ids (List[str]): Unique IDs for each embedding.
            embeddings (List[List[float]]): The actual embedding vectors.
            metadatas (List[Dict]): Associated metadata for each embedding.
        """
        collection = self.get_collection(collection_name)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def reset_collection(self, collection_name:str):
        print(f"Reset collection: {collection_name}")
        try: 
            self.client.delete_collection(collection_name)
        except chromadb.errors.NotFoundError:
            pass

        return self.client.create_collection(collection_name)
