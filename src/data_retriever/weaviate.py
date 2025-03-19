import pandas as pd
import weaviate
from weaviate.classes.config import Configure
import weaviate.classes as wvc
from weaviate.classes.query import Filter

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*'POLLER'.*")

from src.utils import logging, config
from .offers import OffersDatabase
from .embedder import ProductEmbedder

class WeaviateHandler:
    """Handles connection and data management in Weaviate."""

    def __init__(self):
        """Initialize Weaviate connection, model, and SQLite database."""
        try:
            self.collection_name = config["weaviate"]["collection_name"]
            self.db_name = config["database"]["name"]

            self.db = OffersDatabase()
            # self.embedder = ProductEmbedder()

            logging.info("Connecting to Weaviate...")
            self.client = weaviate.connect_to_local()
            if not self.client:
                logging.error("Failed to connect to Weaviate.")
                raise
            
            logging.info("Connected to Weaviate.")

        except Exception as e:
            logging.error(f"Error initializing WeaviateSearch: {e}")
            self.client = None
            self.collection = None
            # self.embedder.model = None


    def create_schema(self):
        """Define and create the schema in Weaviate (v4 syntax)."""
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
                        vectorizer="text2vec-transformers",
                    ),
                    wvc.config.Property(
                        name="description",
                        data_type=wvc.config.DataType.TEXT,
                        vectorizer="text2vec-transformers",
                    ),
                    wvc.config.Property(
                        name="link",
                        data_type=wvc.config.DataType.TEXT,
                    ),
                    wvc.config.Property(
                        name="price",
                        data_type=wvc.config.DataType.NUMBER,
                    ),
                    wvc.config.Property(
                        name="categories",
                        data_type=wvc.config.DataType.TEXT,
                        vectorizer="text2vec-transformers",
                    ),
                    wvc.config.Property(
                        name="rating",
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
                    ),
                ]
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

            for _, row in df.iterrows():
                product = {
                    "product_id": None if pd.isna(row["id"]) else str(row["id"]),
                    "title": None if pd.isna(row["title"]) else str(row["title"]),
                    "description": None if pd.isna(row["description"]) else str(row["description"]),
                    "link": None if pd.isna(row["link"]) else str(row["link"]),
                    "price": None if pd.isna(row["price"]) else row["price"],
                    "categories": None if pd.isna(row["categories"]) else row["categories"],
                    "rating": None if pd.isna(row["rating"]) else row["rating"],
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
        :return: List of all items in the database.
        """
        self.collection = self.client.collections.get(self.collection_name)
        if not self.collection:
            logging.error(f"Collection '{self.collection_name}' not found.")
            raise
        logging.info(f"Collection '{self.collection_name}' loaded successfully.")

        try:
            logging.info("Fetching all items from Weaviate...")

            # Fetch all objects with a high limit (Weaviate caps responses)
            response = self.collection.query.fetch_objects(limit=10000)  

            # Extract object properties safely
            items = [obj.properties for obj in response.objects] if response and response.objects else []

            logging.info(f"Retrieved {len(items)} items from Weaviate.")
            return items

        except Exception as e:
            logging.error(f"Error fetching all items: {e}")
            return []


    def _format_results(self, response):
        """Format search results into a structured response as a string."""
        products_str = []
        all_offers = {}

        for index, obj in enumerate(response.objects, 1):
            product_id = obj.properties.get("product_id", "Unknown")
            title = obj.properties.get("title", "No Title")
            link = obj.properties.get("link", "No Link")
            categories = obj.properties.get("categories", "No Category")
            description = obj.properties.get("description", "No Description")
            image_url = obj.properties.get("image", "none")
            price = obj.properties.get("price", "N/A")
            weight = obj.properties.get("weight", "N/A")
            rating = obj.properties.get("rating", "N/A")

            self.db.connect()
            offers = self.db.find_offers_by_product(product_id)
            self.db.close()

            product_str = f"Product {index} - ID: {product_id}\n"
            product_str += f"Title: {title}\n"
            product_str += f"Link: {link}\n"
            product_str += f"Categories: {categories}\n"
            product_str += f"Description: {description}\n"
            product_str += f"Price: ${price}\n"
            product_str += f"Weight: {weight}\n"
            product_str += f"Rating: {rating}\n"
            product_str += f"Image URL: {image_url}\n"


            if offers:
                for offer in offers:
                    offer_id, offer_name, offer_price, _, _, offer_desc, _, _, _, _, product_list_json = offer
                    product_ids = eval(product_list_json)

                    if offer_id not in all_offers:
                        all_offers[offer_id] = {
                            "name": offer_name,
                            "price": offer_price,
                            "description": offer_desc,
                            "products": []
                        }

                    for pid in product_ids:
                        product_response = self.collection.query.fetch_objects(
                            filters=Filter.by_property("product_id").equal(pid)
                        )
                        if product_response.objects:
                            product_obj = product_response.objects[0].properties
                            product_image = product_obj.get("image", "none")
                            product_title = product_obj.get("title", "No Title")
                            all_offers[offer_id]["products"].append({
                                "image": product_image,
                                "title": product_title
                            })
            
            products_str.append(product_str)

        offers_str = ""
        if all_offers:
            offers_str = "Offers:\n"
            for offer in all_offers.values():
                offer_details = f"Offer: {offer['name']} - Price: {offer['price']}\n"
                offer_details += f"Description: {offer['description']}\n"
                for product in offer["products"]:
                    offer_details += f"- {product['title']} (Image: {product['image']})\n"
                offers_str += offer_details
        else:
            offers_str = "No current offers available for the matching products.\n"

        if products_str:
            result_str = "\n".join(products_str) + "\n\n" + offers_str
        else:
            result_str = "No matching items found.\n"

        logging.info(f"Found {len(products_str)} products and {len(all_offers)} offers.")
        return result_str

    def process_and_store_products(self):
        """Reads product data, generates embeddings, and stores in Weaviate."""
        try:
            try:
                df = pd.read_csv(config['input_file']['cleaned_products_data_path'])
                logging.info("File successfully read")
            except UnicodeDecodeError as e:
                logging.error(f"Error: The file is not UTF-8 encoded. Encoding issue: {e}")
                raise

            # df = self.embedder.generate_embeddings(df)
            self.create_schema()
            self.insert_data(df)
        
        except Exception as main_error:
            logging.critical(f"Critical Error: {main_error}")
        
        finally:
            self.close()

    
    def hybrid_search(self, query, alpha=0.5, limit=5, filters=None):
        """Perform hybrid search using keyword & vector similarity."""
        self.collection = self.client.collections.get(self.collection_name)
        if not self.collection:
            logging.error(f"Collection '{self.collection_name}' not found.")
            raise
        logging.info(f"Collection '{self.collection_name}' loaded successfully.")

        # if not self.collection or not self.embedder.model:
        #     logging.error("Weaviate collection or model not initialized.")
        #     return ["Error: Search service unavailable."]
        
        try:
            response = self.collection.query.hybrid(
                query=query,
                alpha=alpha,
                limit=limit,
                target_vector="info_vector",
                filters=filters
            )
            print(self._format_results(response))
        except Exception as e:
            logging.error(f"Hybrid search failed: {e}")
            return ["Error: Hybrid search failed."]

    def keyword_search(self, query, limit=3, filters=None):
        """Perform BM25 keyword search."""
        self.collection = self.client.collections.get(self.collection_name)
        if not self.collection:
            logging.error(f"Collection '{self.collection_name}' not found.")
            raise
        logging.info(f"Collection '{self.collection_name}' loaded successfully.")

        if not self.collection:
            logging.error("Weaviate collection not initialized.")
            return ["Error: Search service unavailable."]
        
        try:
            response = self.collection.query.bm25(
                query=query, limit=limit, filters=filters
            )
            return self._format_results(response)
        except Exception as e:
            logging.error(f"Keyword search failed: {e}")
            return ["Error: Keyword search failed."]

    def close(self):
        """Close Weaviate connection properly."""
        try:
            self.client.close()
            logging.info("Weaviate connection closed.")
        except Exception as e:
            logging.error(f"Error closing Weaviate connection: {e}")
            raise
