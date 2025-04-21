import atexit

from src.llm import ChatbotHandler
from src.data_retriever import WeaviateHandler
from src.web_scraping import ProductScraper, ProductCleaner
from src.utils import logging

should_update_data = True


def on_shutdown(chatbot_handler):
    """Shutdown function to properly close resources when the program ends."""
    try:
        logging.info("Shutting down chatbot handler...")
        chatbot_handler.shutdown()
        logging.info("Shutdown complete.")
    except Exception as e:
        logging.error(f"Error during shutdown: {e}")


def update_and_create_vector_db(weaviate_handler):
    try:
        logging.info("Starting data update and vector DB creation...")
        
        scraper = ProductScraper()
        cleaner = ProductCleaner()

        logging.info("Running web scraper...")
        scraper.run()
        logging.info("Web scraping completed.")

        logging.info("Expanding and saving product data...")
        cleaner.expand_and_save()
        logging.info("Data expansion completed.")

        logging.info("Processing and storing products into Weaviate...")
        weaviate_handler.process_and_store_products()
        logging.info("Vector DB update completed successfully.")

    except Exception as e:
        logging.error(f"Error while updating and creating vector DB: {e}")


def run_chatbot(weaviate_handler):
    logging.info("Launching chatbot interface...")
    try:
        chatbot_handler = ChatbotHandler(weaviate_handler)
        chatbot_handler.launch_chatbot()
        atexit.register(on_shutdown, chatbot_handler)
    except Exception as e:
        logging.error(f"Fatal error in chatbot: {e}")


def main():
    global should_update_data

    weaviate_handler = WeaviateHandler()
    try:
        if should_update_data:
            update_and_create_vector_db(weaviate_handler)
        else:
            logging.info("Skipping data update as per configuration.")

        run_chatbot(weaviate_handler)
    except Exception as e:
        logging.error(f"Unhandled exception in main: {e}")


if __name__ == "__main__":
    main()
