import pandas as pd
import weaviate
from weaviate.classes.config import Configure
import weaviate.classes as wvc
from weaviate.classes.query import Filter

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
            self.embedder = ProductEmbedder()

            # Initialize Weaviate
            logging.info("Connecting to Weaviate...")
            self.client = weaviate.connect_to_local()
            if not self.client:
                logging.error("Failed to connect to Weaviate.")
                raise
            
            logging.info("Connected to Weaviate.")
            self.collection = self.client.collections.get(self.collection_name)
            if not self.collection:
                logging.error(f"Collection '{self.collection_name}' not found.")
                raise

            logging.info(f"Collection '{self.collection_name}' loaded successfully.")

        except Exception as e:
            logging.error(f"Error initializing WeaviateSearch: {e}")
            self.client = None
            self.collection = None
            self.embedder.model = None


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
            collection = self.client.collections.get(self.collection_name)
            logging.info("Inserting data into Weaviate...")

            for _, row in df.iterrows():
                product = {
                    "product_id": None if pd.isna(row["id"]) else str(row["id"]),
                    "title": None if pd.isna(row["title"]) else str(row["title"]),
                    "price": None if pd.isna(row["price"]) else row["price"],
                    "categories": None if pd.isna(row["categories"]) else row["categories"],
                    "rating": None if pd.isna(row["rating"]) else row["rating"],
                    "weight": None if pd.isna(row["weight"]) else row["weight"],
                    "image": None if pd.isna(row["image"]) else row["image"],
                    "stock_status": None if pd.isna(row["stock_status"]) else row["stock_status"]
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
    
    def _format_results(self, response):
        """Format search results into a structured response."""
        results = []
        all_offers = {}

        for index, obj in enumerate(response.objects, 1):
            product_id = obj.properties.get("product_id", "Unknown")
            title = obj.properties.get("title", "No Title")
            categories = obj.properties.get("categories", "No Category")
            image_url = obj.properties.get("image", "none")
            price = obj.properties.get("price", "N/A")
            rating = obj.properties.get("rating", "N/A")

            self.db.connect()
            offers = self.db.find_offers_by_product(product_id)
            self.db.close()
            offers_text = ""

            if offers:
                offers_text = "<p><strong>Special Offers Available:</strong> Check the offers section below.</p>"
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

            result_html = f"""
            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 10px; margin-bottom: 10px;">
                <h1>Item {index}</h1>
                <img src="{image_url}" alt="Product Image" style="width: 200px; height: auto; border-radius: 10px;">
                <h2>{title}</h2>
                <p><strong>Categories:</strong> {categories}</p>
                <p><strong>Price:</strong> ${price}</p>
                <p><strong>Rating:</strong> {rating} ⭐</p>
                {offers_text}
            </div>
            """

            results.append(result_html)

        if all_offers:
            all_offers_html = "<h2>All Available Offers:</h2>"
            for offer in all_offers.values():
                product_images_html = "".join(
                    f"""
                    <div style="text-align: center; margin-right: 15px;display: flex; flex-direction: column; align-items: center; border: 1px solid white; padding-bottom: 5px; border-radius: 10px; width: 100px;">
                        <img src="{prod['image']}" alt="Offer Product" style="width: 100px; height: 100px; border-radius: 8px; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
                        <p style="font-size: 12px; margin-top: 5px;">{prod['title']}</p>
                    </div>
                    """
                    for prod in offer["products"]
                )
                all_offers_html += f"""
                <div style="border: 1px solid #aaa; border-radius: 10px; padding: 10px; margin-top: 10px;">
                    <h3 style="margin-top: 0px;">{offer["name"]} - {offer["price"]}</h3>
                    <p>{offer["description"]}</p>
                    <div style="display: flex; flex-wrap: wrap;">{product_images_html}</div>
                </div>
                """

            results.append(all_offers_html)

        logging.info(f"Found {len(results)} results, including {len(all_offers)} offers")
        return results if results else ["No matching items found."]
    
    def hybrid_search(self, query, alpha=0.5, limit=5, filters=None):
        """Perform hybrid search using keyword & vector similarity."""
        if not self.collection or not self.embedder.model:
            logging.error("Weaviate collection or model not initialized.")
            return ["Error: Search service unavailable."]
        
        try:
            query_embedding = self.embedder.model.encode(query).tolist()
            response = self.collection.query.hybrid(
                query=query,
                vector=query_embedding,
                alpha=alpha,
                limit=limit,
                target_vector="info_vector",
                filters=filters
            )
            return self._format_results(response)
        except Exception as e:
            logging.error(f"Hybrid search failed: {e}")
            return ["Error: Hybrid search failed."]

    def keyword_search(self, query, limit=3, filters=None):
        """Perform BM25 keyword search."""
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
