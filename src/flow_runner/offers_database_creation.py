from src.data_retriever import OffersDatabase
from src.utils import logging

if __name__ == "__main__":
    try:
        db = OffersDatabase()
        
        db.connect()
        db.create_table() 
        db.insert_data()

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        
    finally:
        db.close()
