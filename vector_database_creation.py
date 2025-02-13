import pandas as pd
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Configure
import weaviate
import weaviate.classes as wvc
import numpy as np
import logging
from datetime import datetime
import os

os.makedirs("./logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    filename=f"./logs/app_log_{datetime.now().strftime('%Y-%m-%d')}.log",
    filemode='a'
)


class ProductEmbedder:
    """Handles embedding generation for product data."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the sentence transformer model."""
        try:
            self.model = SentenceTransformer(model_name)
            logging.info("Embedding model initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing embedding model: {e}")
            raise


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
            self.client.collections.delete(name="Product")
            self.client.collections.create(
                name="Product",
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
            )

            logging.info("Schema created successfully.")
        except Exception as e:
            logging.error(f"Error creating schema: {e}")
            raise


    def clean_data(self, df):
        """Clean the product data to ensure no NaN, inf, or invalid values."""
        for _, row in df.iterrows():
            try:
                if isinstance(row["price"], str):
                    cleaned_price = float(row["price"].replace(' $', '').replace('$', '').replace(',', '').strip())
                else:
                    cleaned_price = float(row["price"])
                
                # Ensure no out-of-range or invalid float values (NaN, inf)
                if cleaned_price == float('inf') or cleaned_price == float('-inf') or np.isnan(cleaned_price):
                    cleaned_price = 0.0
            except ValueError:
                cleaned_price = 0.0

            logging.info(f"Done Cleaning price")
            
            try:
                cleaned_rating = float(str(row["rating"]).strip().replace('"', '').replace("'", ""))
                
                # Ensure no out-of-range or invalid float values
                if cleaned_rating == float('inf') or cleaned_rating == float('-inf') or cleaned_rating != cleaned_rating:  # NaN check
                    cleaned_rating = 0.0
            except ValueError:
                cleaned_rating = 0.0

            logging.info(f"Done Cleaning rating")

            if (np.isnan(cleaned_price) or np.isinf(cleaned_price) or np.isnan(cleaned_rating) or np.isinf(cleaned_rating)):
                logging.error(f"Invalid price or rating for product {row['id']}. Skipping this entry.")
                continue

            df.at[_, "price"] = cleaned_price
            df.at[_, "rating"] = cleaned_rating

        logging.info("Data cleaned successfully.")
        return df


    def insert_data(self, df):
        """Insert product data into Weaviate."""
        try:
            collection = self.client.collections.get("Product")  # Get the Product collection
            logging.info("Inserting data into Weaviate...")

            for _, row in df.iterrows():
                # Construct the product object
                product = {
                    "product_id": str(row["id"]),
                    "title": row["title"],
                    "price": row["price"],
                    "categories": row["categories"],
                    "description": row["description"],
                    "rating": row["rating"],
                    "weight": row["weight"],
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
            df = pd.read_csv("products.csv")
        except UnicodeDecodeError as e:
            logging.error(f"Error: The file is not UTF-8 encoded. Encoding issue: {e}")
            raise

        # Generate embeddings
        embedder = ProductEmbedder()
        df = embedder.generate_embeddings(df)

        # Connect to Weaviate and manage data
        weaviate_handler = WeaviateHandler()
        df = weaviate_handler.clean_data(df)
        weaviate_handler.create_schema()
        weaviate_handler.insert_data(df)

    except Exception as main_error:
        logging.critical(f"Critical Error: {main_error}")

    finally:
        weaviate_handler.close_connection()