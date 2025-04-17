import sqlite3
import pandas as pd
import json
import os

from src.utils import logging, config


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

    def insert_data(self, csv_file=config['data_file']['offers_data_path']):
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
    
    def find_offers_by_product(self, product_id):
        """Find offers that contain a given product ID."""
        try:
            cursor = self.conn.cursor()
            query = """
                SELECT * FROM offers
                WHERE ? IN (SELECT value FROM json_each(product_list))
            """
            cursor.execute(query, (str(product_id),))
            results = cursor.fetchall()
            return results
        except Exception as e:
            logging.error(f"Error fetching offers for product {product_id}: {e}")
            return []
    
    def get_offers(self, limit=20):
        """Retrieve a limited number of offers from the database and return them as strings."""
        self.connect()
        try:
            cursor = self.conn.cursor()
            query = f"""
                SELECT id, title, price, link, categories, description, rating, weight, image, stock_status 
                FROM offers
                ORDER BY id DESC
                LIMIT ?
            """
            cursor.execute(query, (limit,))
            results = cursor.fetchall()

            offer_strings = "\n\n".join(
            f"Offer {index + 1}: Title: {row[1]}, Price: {row[2]}, Link: {row[3]}, Categories: {row[4]}, "
            f"Description: {row[5]}, Rating: {row[6]}, Weight: {row[7]}, Image: {row[8]}, Stock Status: {row[9]}"
            for index, row in enumerate(results)
            )

            self.close()            
            return offer_strings
        except sqlite3.Error as e:
            logging.error(f"Error fetching offers: {e}")
            self.close() 
            return []


    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed.")

