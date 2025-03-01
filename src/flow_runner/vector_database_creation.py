import pandas as pd

from ..utils.logger_setup import logging
from ..utils.config_loader import config
from ..data_retriever.weaviate import WeaviateHandler
from ..data_retriever.embedder import ProductEmbedder

def process_and_store_products():
    """Reads product data, generates embeddings, and stores in Weaviate."""
    try:
        try:
            df = pd.read_csv(config['input_file']['cleaned_products_data_path'])
            logging.info("File successfully read")
        except UnicodeDecodeError as e:
            logging.error(f"Error: The file is not UTF-8 encoded. Encoding issue: {e}")
            raise

        embedder = ProductEmbedder()
        df = embedder.generate_embeddings(df)

        weaviate_handler = WeaviateHandler()
        weaviate_handler.create_schema()
        weaviate_handler.insert_data(df)
    
    except Exception as main_error:
        logging.critical(f"Critical Error: {main_error}")
    
    finally:
        weaviate_handler.close()

if __name__ == "__main__":
    process_and_store_products()
