from src.data_retriever import ProductDataCleaner
from src.utils import logging

if __name__ == "__main__":
    try:
        cleaner = ProductDataCleaner()
        cleaned_file = cleaner.process_file()
        logging.info("File processed successfully.")
    
    except Exception as e:
        logging.error(f"Error during file processing: {e}")
