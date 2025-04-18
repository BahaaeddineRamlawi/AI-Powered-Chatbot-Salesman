from src.web_scraping import ProductCleaner
from src.utils import logging

if __name__ == "__main__":
    try:
        cleaner = ProductCleaner()
        logging.info("Expanding, cleaning and saving product data...")
        cleaner.expand_and_save()
        logging.info("Data expansion completed.")
    except Exception as e:
        logging.error(f"Error during expanding and cleaning data: {e}")
