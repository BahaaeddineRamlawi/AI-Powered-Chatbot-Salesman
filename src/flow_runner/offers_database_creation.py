from src.data_retriever import OffersDatabase


if __name__ == "__main__":
    db = OffersDatabase()
    db.connect()
    db.create_table()
    db.insert_data()
    db.close()
