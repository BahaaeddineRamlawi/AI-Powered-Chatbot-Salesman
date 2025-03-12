from src.utils import config
from src.data_retriever import ProductDataCleaner

if __name__ == "__main__":
    cleaner = ProductDataCleaner(config['input_file']['products_data_path'])
    cleaned_file = cleaner.process_file()