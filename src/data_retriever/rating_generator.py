import random
import pandas as pd

from src.utils import logging, config

class UserRatingsGenerator:
    def __init__(self, num_users=20):
        self.input_csv = config['data_file']['cleaned_products_data_path']
        self.output_csv = config['data_file']['user_rating_datapath']
        self.target_ratings = num_users * 125
        self.num_users = num_users
        self.user_ids = [f"{i}" for i in range(1, num_users + 1)]
        self.product_ids = []
        self.ratings_data = []

    def load_products(self):
        """Load product IDs from the input CSV file."""
        try:
            df = pd.read_csv(self.input_csv)
            self.product_ids = df['product_id'].tolist()
            logging.info(f"Loaded {len(self.product_ids)} products.")
        except Exception as e:
            logging.error(f"Failed to load product data: {e}")
            raise

    def generate_ratings(self):
        """Generate random ratings per user."""
        try:
            for user in self.user_ids:
                num_ratings = random.randint(100, 150)
                if len(self.ratings_data) + num_ratings > self.target_ratings:
                    num_ratings = self.target_ratings - len(self.ratings_data)

                products_to_rate = random.sample(self.product_ids, num_ratings)
                for product in products_to_rate:
                    rating = random.randint(1, 5)
                    self.ratings_data.append({
                        "user_id": user,
                        "product_id": product,
                        "rating": rating
                    })

                logging.info(f"Generated {num_ratings} ratings for user {user}.")

                if len(self.ratings_data) >= self.target_ratings:
                    logging.info("Reached target rating count.")
                    break
        except Exception as e:
            logging.error(f"Error during rating generation: {e}")
            raise

    def save_to_csv(self):
        """Save the generated ratings to the output CSV file."""
        try:
            df = pd.DataFrame(self.ratings_data)
            df.to_csv(self.output_csv, index=False)
            logging.info(f"Saved {len(self.ratings_data)} ratings to {self.output_csv}.")
        except Exception as e:
            logging.error(f"Failed to save ratings: {e}")
            raise

    def run(self):
        """Run the full rating generation process."""
        logging.info("Starting rating generation process...")
        self.load_products()
        self.generate_ratings()
        self.save_to_csv()
        logging.info("Rating generation process completed successfully.")