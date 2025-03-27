from src.data_retriever import WeaviateHandler
from src.utils import logging

if __name__ == "__main__":
    try:
        weaviate_handler = WeaviateHandler()
        weaviate_handler.process_and_store_products()
    
    except Exception as e:
        logging.error(f"An error occurred during the product processing and storage: {e}")
