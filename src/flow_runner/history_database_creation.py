from src.data_retriever import UserHistoryDatabase
from src.utils import logging

if __name__ == "__main__":
    try:
        db = UserHistoryDatabase()
        
        db.connect()
        db.create_table()
        db.insert_data()
    
    except Exception as e:
        logging.error(f"Error in database operations: {e}")
    
    finally:
        db.close()
        logging.info("Database connection closed.")
