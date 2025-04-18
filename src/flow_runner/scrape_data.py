from src.web_scraping import ProductScraper
from src.utils import logging

if __name__ == "__main__":
    try:
        scraper = ProductScraper()
        logging.info("Running web scraper...")
        scraper.run()
        logging.info("Web scraping completed.")
    except Exception as e:
        logging.error(f"Error during scraping data: {e}")
