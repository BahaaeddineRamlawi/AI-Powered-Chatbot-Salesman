from src.data_retriever import UserHistoryDatabase
from src.utils import logging

if __name__ == "__main__":
    try:
        db = UserHistoryDatabase()
        db.connect()
        logging.info("Database connected successfully.")
        
        db.create_table()
        logging.info("Table created successfully.")
        
        db.insert_data()
        logging.info("User history data inserted successfully.")
    
    except Exception as e:
        logging.error(f"Error in database operations: {e}")
    
    finally:
        db.close()
        logging.info("Database connection closed.")
