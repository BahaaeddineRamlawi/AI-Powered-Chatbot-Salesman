from ..utils.config_loader import config
from ..data_retriever.data_cleaner import ProductDataCleaner

if __name__ == "__main__":
    cleaner = ProductDataCleaner(config['input_file']['products_data_path'])
    cleaned_file = cleaner.process_file()