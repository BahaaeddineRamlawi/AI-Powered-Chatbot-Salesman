from src.data_retriever import UserRatingsGenerator
from src.utils import logging

if __name__ == "__main__":
    try:
        logging.info("Initializing UserRatingsGenerator...")
        generator = UserRatingsGenerator()
        logging.info("Running the rating generation process...")
        generator.run()
        logging.info("Rating generation completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred while generating user ratings: {e}")
