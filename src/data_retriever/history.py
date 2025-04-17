import sqlite3
import os
import json
import pandas as pd
from datetime import datetime

from src.utils import logging, config

class UserHistoryDatabase:
    def __init__(self, db_name=config['database']['history_name']):
        """Initialize the database connection."""
        self.db_name = db_name
        self.conn = None
        self.db_dir = os.path.dirname(self.db_name)
        if self.db_dir and not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir, exist_ok=True)

    def connect(self):
        """Establish a connection to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_name)
            logging.info("Connected to the SQLite user history database.")
        except sqlite3.Error as e:
            logging.error(f"Error connecting to database: {e}")

    def create_table(self):
        """Create the user history table with product lists and timestamps."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE,
                    product_list TEXT,
                    latest_timestamp TEXT
                )
            """)
            self.conn.commit()
            logging.info("User history table created successfully or already exists.")
        except sqlite3.Error as e:
            logging.error(f"Error creating table: {e}")

    def insert_data(self):
        """Insert user product ratings as a JSON list of (product_id, rating, timestamp) pairs."""
        try:
            df = pd.read_csv(config['data_file']['user_rating_datapath'])

            required_columns = {"user_id", "product_id", "rating"}
            if not required_columns.issubset(df.columns):
                logging.error("CSV file is missing required columns: user_id, product_id, rating")
                return

            current_timestamp = datetime.utcnow().isoformat()

            df["timestamp"] = current_timestamp
            user_ratings = (
                df.groupby("user_id")[["product_id", "rating", "timestamp"]]
                .apply(lambda x: json.dumps(list(x.itertuples(index=False, name=None))))
                .reset_index()
                .rename(columns={0: "product_list"})
            )

            user_ratings["latest_timestamp"] = current_timestamp

            user_ratings.to_sql("user_history", self.conn, if_exists="replace", index=False)
            self.conn.commit()
            logging.info("User product ratings inserted successfully.")
        except Exception as e:
            logging.error(f"Error inserting data: {e}")
    
    def get_user_products(self, user_id):
        """Retrieve the list of (product_id, rating) pairs for a given user ID."""
        try:
            cursor = self.conn.cursor()
            query = "SELECT product_list FROM user_history WHERE user_id = ?"
            cursor.execute(query, (user_id,))
            result = cursor.fetchone()

            if result:
                product_list = json.loads(result[0])
                return [(item[0], item[1]) for item in product_list]
            else:
                logging.info(f"No data found for user_id {user_id}.")
                return []
        except sqlite3.Error as e:
            logging.error(f"Error fetching products for user {user_id}: {e}")
            return []
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON for user {user_id}: {e}")
            return []

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logging.info("User history database connection closed.")
