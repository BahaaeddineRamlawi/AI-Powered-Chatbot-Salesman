import atexit
import os
import time
from datetime import datetime, timedelta

from src.llm import ChatbotHandler
from src.data_retriever import WeaviateHandler, UserRatingsGenerator, UserHistoryDatabase
from src.web_scraping import ProductScraper, ProductCleaner
from src.utils import logging, config


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

        with open(config["data_file"]["last_updated_file"], "w") as f:
            f.write(str(int(time.time())))

    except Exception as e:
        logging.error(f"Error while updating and creating vector DB: {e}")


def should_update():
    """Check if data update is needed based on the last update timestamp."""
    if not os.path.exists(config["data_file"]["last_updated_file"]):
        logging.info("No update timestamp found. Proceeding with update.")
        return True

    try:
        with open(config["data_file"]["last_updated_file"], "r") as f:
            last_timestamp = int(f.read().strip())
            last_update_time = datetime.fromtimestamp(last_timestamp)
            next_update_time = last_update_time + timedelta(days=7)

            if datetime.now() >= next_update_time:
                logging.info(f"More than {7} days since last update. Proceeding.")
                return True
            else:
                logging.info(f"Last update was on {last_update_time}. Skipping update.")
                return False
    except Exception as e:
        logging.warning(f"Failed to read or parse last update file. Proceeding with update: {e}")
        return True

def manage_user_history_db():
    try:
        logging.info("Initializing UserHistoryDatabase...")
        db = UserHistoryDatabase()
        db.connect()
        db.create_table()
        db.insert_data()
        logging.info("User history database operations completed successfully.")
    except Exception as e:
        logging.error(f"Error in database operations: {e}")
    finally:
        db.close()
        logging.info("Database connection closed.")

def generate_user_ratings():
    try:
        logging.info("Initializing UserRatingsGenerator...")
        generator = UserRatingsGenerator()
        generator.run()
        logging.info("Rating generation completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred while generating user ratings: {e}")


def run_chatbot(weaviate_handler):
    logging.info("Launching chatbot interface...")
    try:
        chatbot_handler = ChatbotHandler(weaviate_handler)
        chatbot_handler.launch_chatbot()
        atexit.register(on_shutdown, chatbot_handler)
    except Exception as e:
        logging.error(f"Fatal error in chatbot: {e}")


def main():
    weaviate_handler = WeaviateHandler()

    try:
        if should_update():
            update_and_create_vector_db(weaviate_handler)
            # generate_user_ratings()
            # manage_user_history_db()
        else:
            logging.info("Skipping data update (already up-to-date).")

        run_chatbot(weaviate_handler)

    except Exception as e:
        logging.error(f"Unhandled exception in main: {e}")


if __name__ == "__main__":
    main()
