import pandas as pd
import weaviate
from weaviate.classes.config import Configure
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.utils import logging, config
from .reranker import RerankedResponse

class WeaviateHandler:
    """Handles connection and data management in Weaviate."""

    def __init__(self):
        """Initialize Weaviate connection, model, and SQLite database."""
        try:
            logging.info("Initializing WeaviateSearch...")
            self.collection_name = config["weaviate"]["collection_name"]
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            self.reranked = RerankedResponse(self.embedding_model)
            logging.info("Ranker initialized.")

            self.client = weaviate.connect_to_local()

            if not self.client:
                logging.error("Failed to connect to Weaviate. Client is None.")
                raise Exception("Weaviate connection failed. Client is None.")
            
            logging.info("Successfully connected to Weaviate.")

        except Exception as e:
            logging.error(f"Error during initialization of WeaviateSearch: {e}")
            self.client = None
            self.collection = None

            if self.client:
                logging.info("Closing Weaviate client.")
                self.client.close()
            
            logging.info("WeaviateSearch initialization failed.")


    def create_schema(self):
        """Define and create the schema in Weaviate (v4 syntax)."""
        embedding_model = config["embedding"]["model_type"]
        try:
            self.client.collections.delete(name=self.collection_name)
            self.client.collections.create(
                name=self.collection_name,
                properties=[
                    wvc.config.Property(
                        name="product_id",
                        data_type=wvc.config.DataType.TEXT,
                    ),
                    wvc.config.Property(
                        name="title",
                        data_type=wvc.config.DataType.TEXT,
                        vectorizer=embedding_model,
                    ),
                    wvc.config.Property(
                        name="description",
                        data_type=wvc.config.DataType.TEXT,
                        vectorizer=embedding_model,
                    ),
                    wvc.config.Property(
                        name="link",
                        data_type=wvc.config.DataType.TEXT,
                    ),
                    wvc.config.Property(
                        name="price",
                        data_type=wvc.config.DataType.NUMBER,
                        vectorizer=embedding_model,
                    ),
                    wvc.config.Property(
                        name="categories",
                        data_type=wvc.config.DataType.TEXT_ARRAY,
                        vectorizer=embedding_model,
                    ),
                    wvc.config.Property(
                        name="rating",
                        data_type=wvc.config.DataType.NUMBER,
                    ),
                    wvc.config.Property(
                        name="rating_count",
                        data_type=wvc.config.DataType.NUMBER,
                    ),
                    wvc.config.Property(
                        name="weight",
                        data_type=wvc.config.DataType.TEXT,
                    ),
                    wvc.config.Property(
                        name="image",
                        data_type=wvc.config.DataType.TEXT,
                    ),
                    wvc.config.Property(
                        name="stock_status",
                        data_type=wvc.config.DataType.TEXT,
                    ),
                ],
                vectorizer_config=[
                    Configure.NamedVectors.text2vec_transformers(
                        name="info_vector",
                        vector_index_config=Configure.VectorIndex.hnsw()
                    ) if embedding_model == "text2vec-transformers" else None,
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
        self.collection = self.client.collections.get(self.collection_name)
        logging.info(f"Collection '{self.collection_name}' loaded successfully.")

        try:
            logging.info("Inserting data into Weaviate...")

            BATCH_SIZE = 100
            for start in range(0, len(df), BATCH_SIZE):
                batch = df.iloc[start:start + BATCH_SIZE]
                for _, row in batch.iterrows():
                    categories = None if pd.isna(row["categories"]) else [cat.strip() for cat in row["categories"].split(",")]
                    product = {
                        "product_id": None if pd.isna(row["product_id"]) else str(row["product_id"]),
                        "title": None if pd.isna(row["title"]) else str(row["title"]),
                        "description": None if pd.isna(row["description"]) else str(row["description"]),
                        "link": None if pd.isna(row["link"]) else str(row["link"]),
                        "price": None if pd.isna(row["price"]) else row["price"],
                        "categories": categories,
                        "rating": None if pd.isna(row["rating"]) else row["rating"],
                        "rating_count": None if pd.isna(row["rating_count"]) else row["rating_count"],
                        "weight": None if pd.isna(row["weight"]) else row["weight"],
                        "image": None if pd.isna(row["image"]) else row["image"],
                        "stock_status": None if pd.isna(row["stock_status"]) else row["stock_status"]
                    }
                    self.collection.data.insert(
                        properties=product,
                    )

            logging.info("Data insertion complete.")
        except Exception as e:
            logging.error(f"Error inserting data: {e}")
            raise
        

    def get_all_items(self):
        """
        Retrieve all items from the Weaviate collection.
        """
        self.collection = self.client.collections.get(self.collection_name)
        if not self.collection:
            logging.error(f"Collection '{self.collection_name}' not found.")
            raise
        logging.info(f"Collection '{self.collection_name}' loaded successfully.")

        try:
            response = self.collection.query.fetch_objects(limit=10000)  

            items = [obj.properties for obj in response.objects] if response and response.objects else []

            logging.info(f"Retrieved {len(items)} items from Weaviate.")
            return items

        except Exception as e:
            logging.error(f"Error fetching all items: {e}")
            return []


    def _format_results(self, response):
        """Format search results into a structured response as a single string."""
        result = ""

        for index, obj in enumerate(response.objects, 1):
            title = obj.properties.get("title", "No Title")
            link = obj.properties.get("link", "No Link")
            categories = obj.properties.get("categories", "No Category")
            description = obj.properties.get("description", "No Description").rstrip()
            image_url = obj.properties.get("image", "none")
            price = obj.properties.get("price", "N/A")
            weight = obj.properties.get("weight", "N/A")
            rating = obj.properties.get("rating", "N/A")
            rating_count = obj.properties.get("rating_count", "N/A")

            result += (
                f"\nProduct {index}\n"
                f"Title: {title}\n"
                f"Link: {link}\n"
                f"Categories: {categories}\n"
                f"Description: {description}\n"
                f"Price: ${price}\n"
                f"Weight: {weight}\n"
                f"Rating: {rating}\n"
                f"Rating Count: {rating_count}\n"
                f"Image URL: {image_url}\n"
            )

        if not result:
            return "No matching items found.\n"

        logging.info("Result formatted successfully")
        return result


    def process_and_store_products(self):
        """Reads product data, generates embeddings, and stores in Weaviate."""
        try:
            try:
                df = pd.read_csv(config['data_file']['cleaned_products_data_path'])
                logging.info("File successfully read")
            except UnicodeDecodeError as e:
                logging.error(f"Error: The file is not UTF-8 encoded. Encoding issue: {e}")
                raise

            self.create_schema()
            self.insert_data(df)

            logging.info("Product processing and storage completed successfully.")
        
        except Exception as main_error:
            logging.critical(f"Critical Error: {main_error}")
        
        finally:
            self.close()


    def hybrid_search(self, query, alpha=0.5, limit=8, filters=None):
        """Perform hybrid search using keyword & vector similarity."""
        self.collection = self.client.collections.get(self.collection_name)
        
        if not self.collection:
            logging.error(f"Collection '{self.collection_name}' not found.")
            raise
        
        logging.info(f"Collection '{self.collection_name}' loaded successfully.")

        try:
            response = self.collection.query.hybrid(
                query=query,
                alpha=alpha,
                target_vector="info_vector",
                filters=filters
            )
            
            documents = [obj.properties for obj in response.objects]

            if not documents:
                logging.info("Hybrid search returned no results. Skipping reranking.")
                return "No Product Available or Requested", {}
            
            reranked_docs = self.reranked.rerank_results(query, documents)
            self.reranked.process_objects(reranked_docs, limit=limit)

            first_product = reranked_docs[0] if reranked_docs else None
            
            return self._format_results(self.reranked), first_product
            
        except Exception as e:
            logging.error(f"Hybrid search failed: {e}")
            raise


    def close(self):
        """Close Weaviate connection properly."""
        try:
            if self.client:
                self.client.close()
            logging.info("Weaviate connection closed.")
        except Exception as e:
            logging.error(f"Error closing Weaviate connection: {e}")
            raise
