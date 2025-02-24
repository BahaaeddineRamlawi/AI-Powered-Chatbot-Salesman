import sqlite3
import pandas as pd
import json
import logging
from datetime import datetime
import yaml
import os

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

os.makedirs(config['logging']['logs_dir'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    filename=f"{config['logging']['logs_dir']}/app_log_{datetime.now().strftime('%Y-%m-%d')}.log",
    filemode='a'
)


class OffersDatabase:
    def __init__(self, db_name=config['database']['name']):
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
            logging.info("Connected to the SQLite database.")
        except sqlite3.Error as e:
            logging.error(f"Error connecting to database: {e}")

    def create_table(self):
        """Create the offers table if it does not exist."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS offers (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    price REAL,
                    link TEXT,
                    categories TEXT,
                    description TEXT,
                    rating REAL,
                    weight TEXT,
                    image TEXT,
                    stock_status TEXT,
                    product_list TEXT
                )
            """)
            self.conn.commit()
            logging.info("Table created successfully or already exists.")
        except sqlite3.Error as e:
            logging.error(f"Error creating table: {e}")

    def insert_data(self, csv_file=config['input_file']['offers_data_path']):
        """Insert data from a CSV file into the database."""
        try:
            df = pd.read_csv(csv_file)
            
            if "product_list" in df.columns:
                df["product_list"] = df["product_list"].apply(lambda x: json.dumps(x.strip("[]").split(",")))

            df.to_sql("offers", self.conn, if_exists="replace", index=False)
            self.conn.commit()
            logging.info("Data inserted successfully into the database.")
        except Exception as e:
            logging.error(f"Error inserting data: {e}")

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed.")


if __name__ == "__main__":
    db = OffersDatabase()
    db.connect()
    db.create_table()
    db.insert_data()
    db.close()
