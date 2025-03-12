import sqlite3
import json
import os

from src.utils import logging, config

class UserHistoryDatabase:
    def __init__(self, db_name=config['database']['user_history_name']):
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
        """Create the user history table if it does not exist."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    chat_id TEXT,
                    timestamp TEXT,
                    user_message TEXT,
                    chatbot_response TEXT,
                    product_list TEXT
                )
            """
            )
            self.conn.commit()
            logging.info("User history table created successfully or already exists.")
        except sqlite3.Error as e:
            logging.error(f"Error creating table: {e}")
    
    def insert_data(self, data):
        """Insert multiple chat entries into the database."""
        try:
            cursor = self.conn.cursor()
            cursor.executemany("""
                INSERT INTO user_history (user_id, chat_id, timestamp, user_message, chatbot_response, product_list)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [(d["user_id"], d["chat_id"], d["timestamp"], d["user_message"], d["chatbot_response"], json.dumps(d["product_list"])) for d in data])
            self.conn.commit()
            logging.info("Multiple chat entries inserted successfully.")
        except sqlite3.Error as e:
            logging.error(f"Error inserting chat entries: {e}")
    
    def place_item(self, user_id, chat_id, timestamp, user_message, chatbot_response, product_list):
        """Insert a new chat entry into the database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO user_history (user_id, chat_id, timestamp, user_message, chatbot_response, product_list)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, chat_id, timestamp, user_message, chatbot_response, json.dumps(product_list)))
            self.conn.commit()
            logging.info("Chat entry inserted successfully.")
        except sqlite3.Error as e:
            logging.error(f"Error inserting chat entry: {e}")
    
    def get(self, user_id):
        """Retrieve chat history for a given user ID."""
        try:
            cursor = self.conn.cursor()
            query = "SELECT * FROM user_history WHERE user_id = ? ORDER BY timestamp DESC"
            cursor.execute(query, (user_id,))
            results = cursor.fetchall()
            return results
        except sqlite3.Error as e:
            logging.error(f"Error fetching history for user {user_id}: {e}")
            return []

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logging.info("User history database connection closed.")
