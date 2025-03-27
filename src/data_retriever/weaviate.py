import pandas as pd
import weaviate
from weaviate.classes.config import Configure
import weaviate.classes as wvc
from weaviate.classes.query import Filter
from flashrank import Ranker

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.utils import logging, config
from .offers import OffersDatabase
from .reranker import RerankedResponse

class WeaviateHandler:
    """Handles connection and data management in Weaviate."""

    def __init__(self):
        """Initialize Weaviate connection, model, and SQLite database."""
        try:
            logging.info("Initializing WeaviateSearch...")

            self.collection_name = config["weaviate"]["collection_name"]
            self.db_name = config["database"]["name"]

            self.db = OffersDatabase()
            logging.info("Database connection established.")

            self.ranker = Ranker()
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
                    ),
                    wvc.config.Property(
                        name="categories",
                        data_type=wvc.config.DataType.TEXT,
                        vectorizer=embedding_model,
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
        """Format search results into a structured response as a string."""
        products_str = []
        all_offers = {}
        product_ids_to_fetch = []
        offers_by_product = {}
        offers_str = ""

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

            product_ids_to_fetch.append(product_id)

            product_str = f"Product {index}\n"
            product_str += f"Title: {title}\n"
            product_str += f"Link: {link}\n"
            product_str += f"Categories: {categories}\n"
            product_str += f"Description: {description}\n"
            product_str += f"Price: ${price}\n"
            product_str += f"Weight: {weight}\n"
            product_str += f"Rating: {rating}\n"
            product_str += f"Image URL: {image_url}\n"

            products_str.append(product_str)
        
        self.db.connect()
        
        
        for product_id in product_ids_to_fetch:
            offers = self.db.find_offers_by_product(product_id)
            if offers:
                offers_by_product[product_id] = offers
        
        self.db.close()

        for product_id, offers in offers_by_product.items():
            for offer in offers:
                offer_id, offer_name, offer_price, _, _, offer_desc, _, _, _, _, product_list_json = offer
                try:
                    product_ids = eval(product_list_json)
                except Exception as e:
                    logging.error(f"Error processing product_list_json for offer_id {offer_id}: {e}")
                    continue

                all_offers = self.add_offer_to_all_offers(offer_id, offer_name, offer_price, offer_desc, product_ids, all_offers)

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
    
    
    def add_offer_to_all_offers(self, offer_id, offer_name, offer_price, offer_desc, product_ids, all_offers):
        """Adds the offer and its associated products to the all_offers dictionary"""
        try:
            if offer_id not in all_offers:
                all_offers[offer_id] = {
                    "name": offer_name,
                    "price": offer_price,
                    "description": offer_desc,
                    "products": []
                }


            for pid in product_ids:
                logging.info(f"Fetching product details for Product ID: {pid}")

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


            logging.info(f"Offer {offer_id} processing complete.")
            return all_offers

        except Exception as e:
            logging.error(f"Error adding offer {offer_id}: {e}", exc_info=True)
            return all_offers


    def process_and_store_products(self):
        """Reads product data, generates embeddings, and stores in Weaviate."""
        try:
            try:
                df = pd.read_csv(config['input_file']['cleaned_products_data_path'])
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


    def hybrid_search(self, query, alpha=0.5, limit=5, filters=None):
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
                return {"message": "No relevant results found."}
            
            reranked = RerankedResponse()
            reranked_docs = reranked.rerank_results(query, documents)
            reranked.process_objects(reranked_docs, limit=limit)
            
            return self._format_results(reranked)
            
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
