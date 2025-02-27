import pandas as pd
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Configure
import weaviate
import weaviate.classes as wvc
import numpy as np

from utils import logging, config



class ProductEmbedder:
    """Handles embedding generation for product data."""
    
    def __init__(self):
        """Initialize the embedding model based on the config file."""
        model_type = config['embedding']['model_type'] 
        model_name = config['embedding']['model_name']
        
        if model_type == "sentencetransformer":
            try:
                self.model = SentenceTransformer(model_name)
                logging.info(f"Embedding model '{model_name}' initialized successfully.")
            except Exception as e:
                logging.error(f"Error initializing embedding model: {e}")
                raise
        else:
            logging.error(f"Model type '{model_type}' is not supported yet.")
            raise ValueError(f"Unsupported model type: {model_type}")  # Raise an error


    def generate_embeddings(self, df):
        """Generate and combine embeddings for the title and description."""
        try:
            logging.info("Generating embeddings...")

            df["combined_text"] = df.apply(lambda x: f"title: {x['title']} description: {x['description']} categories: {x['categories']}", axis=1)
            df["info_vector"] = df["combined_text"].apply(lambda x: self.model.encode(x))

            logging.info("Embeddings generated and validated successfully.")
            return df
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            raise


class WeaviateHandler:
    """Handles connection and data management in Weaviate."""

    def __init__(self):
        """Initialize connection to Weaviate."""
        try:
            self.client = weaviate.connect_to_local()
            logging.info("Connected to Weaviate.")
        except Exception as e:
            logging.error(f"Error connecting to Weaviate: {e}")
            raise


    def create_schema(self):
        """Define and create the schema in Weaviate (v4 syntax)."""
        try:
            self.client.collections.delete(name=config['weaviate']['collection_name'])
            self.client.collections.create(
                name=config['weaviate']['collection_name'],
                properties=[
                    wvc.config.Property(
                        name="product_id",
                        data_type=wvc.config.DataType.TEXT,
                        vectorize_property_name=False
                    ),
                    wvc.config.Property(
                        name="title",
                        data_type=wvc.config.DataType.TEXT,
                        vectorize_property_name=False
                    ),
                    wvc.config.Property(
                        name="price",
                        data_type=wvc.config.DataType.NUMBER,
                        vectorize_property_name=False
                    ),
                    wvc.config.Property(
                        name="categories",
                        data_type=wvc.config.DataType.TEXT,
                        vectorize_property_name=False
                    ),
                    wvc.config.Property(
                        name="description",
                        data_type=wvc.config.DataType.TEXT,
                        vectorize_property_name=False
                    ),
                    wvc.config.Property(
                        name="rating",
                        data_type=wvc.config.DataType.NUMBER,
                        vectorize_property_name=False
                    ),
                    wvc.config.Property(
                        name="weight",
                        data_type=wvc.config.DataType.TEXT,
                        vectorize_property_name=False
                    ),
                    wvc.config.Property(
                        name="image",
                        data_type=wvc.config.DataType.TEXT,
                        vectorize_property_name=False
                    ),
                    wvc.config.Property(
                        name="stock_status",
                        data_type=wvc.config.DataType.TEXT,
                        vectorize_property_name=False
                    ),
                ],
                vectorizer_config=[
                    Configure.NamedVectors.none(
                        name="info_vector",
                        vector_index_config=Configure.VectorIndex.hnsw()
                    ),
                ],
                inverted_index_config=Configure.inverted_index(
                    index_null_state=True
                )
            )

            logging.info("Schema created successfully.")
        except Exception as e:
            logging.error(f"Error creating schema: {e}")
            raise


    def insert_data(self, df):
        """Insert product data into Weaviate."""
        try:
            collection = self.client.collections.get(config['weaviate']['collection_name'])
            logging.info("Inserting data into Weaviate...")

            for _, row in df.iterrows():
                product = {
                    "product_id": str(row["id"]),
                    "title": str(row["title"]),
                    "price": None if pd.isna(row["price"]) else row["price"],
                    "categories": row["categories"],
                    # "description": row["description"],
                    "rating": None if pd.isna(row["rating"]) else row["rating"],
                    "weight": None if pd.isna(row["weight"]) else row["weight"],
                    "image": row["image"],
                    "stock_status": row["stock_status"]
                }

                # Add the pre-generated embeddings
                product["info_vector"] = row["info_vector"].tolist() 

                # Insert the data with the embeddings
                collection.data.insert(
                    properties=product,
                )

            logging.info("Data insertion complete.")
        except Exception as e:
            logging.error(f"Error inserting data: {e}")
            raise

    def close_connection(self):
        """Close Weaviate connection properly."""
        try:
            self.client.close()
            logging.info("Weaviate connection closed.")
        except Exception as e:
            logging.error(f"Error closing Weaviate connection: {e}")
            raise


if __name__ == "__main__":
    try:
        # Load CSV
        try:
            df = pd.read_csv(config['input_file']['cleaned_products_data_path'])
            logging.info("File successfully read")
        except UnicodeDecodeError as e:
            logging.error(f"Error: The file is not UTF-8 encoded. Encoding issue: {e}")
            raise

        # Generate embeddings
        embedder = ProductEmbedder()
        df = embedder.generate_embeddings(df)

        # Connect to Weaviate and manage data
        weaviate_handler = WeaviateHandler()
        weaviate_handler.create_schema()
        weaviate_handler.insert_data(df)

    except Exception as main_error:
        logging.critical(f"Critical Error: {main_error}")

    finally:
        weaviate_handler.close_connection()
