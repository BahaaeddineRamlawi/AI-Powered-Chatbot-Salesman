import weaviate
import gradio as gr
from sentence_transformers import SentenceTransformer
from weaviate.classes.query import Filter
import sqlite3

from utils import logging, config

class WeaviateSearch:
    def __init__(self):
        """Initialize Weaviate connection, model, and SQLite database."""
        try:
            self.collection_name = config["weaviate"]["collection_name"]
            self.model_name = config["embedding"]["model_name"]
            self.model_type = config["embedding"]["model_type"]
            self.db_name = config["database"]["name"]

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

            # Load SentenceTransformer model
            if self.model_type.lower() == "sentencetransformer":
                try:
                    self.model = SentenceTransformer(self.model_name)
                    logging.info(f"Model '{self.model_name}' loaded successfully.")
                except Exception as e:
                    logging.error(f"Error loading embedding model: {e}")
                    raise RuntimeError(f"Failed to initialize embedding model: {e}")
            else:
                logging.error(f"Model type '{self.model_type}' is not supported yet.")
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            logging.info(f"Model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logging.error(f"Error initializing WeaviateSearch: {e}")
            self.client = None
            self.collection = None
            self.model = None


    def find_offers_by_product(self, product_id):
        """Find offers that contain a given product ID."""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            query = """
                SELECT * FROM offers
                WHERE ? IN (SELECT value FROM json_each(product_list))
            """
            cursor.execute(query, (str(product_id),))
            results = cursor.fetchall()
            conn.close()
            return results
        except Exception as e:
            logging.error(f"Error fetching offers for product {product_id}: {e}")
            return []

    def hybrid_search(self, query, alpha=0.5, limit=5, filters=None):
        """Perform hybrid search using keyword & vector similarity."""
        if not self.collection or not self.model:
            logging.error("Weaviate collection or model not initialized.")
            return ["Error: Search service unavailable."]
        
        try:
            query_embedding = self.model.encode(query).tolist()
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

            offers = self.find_offers_by_product(product_id)
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



    def close(self):
        """Close Weaviate connection."""
        if self.client:
            try:
                self.client.close()
                logging.info("Weaviate connection closed.")
            except Exception as e:
                logging.error(f"Error closing Weaviate connection: {e}")

search_engine = WeaviateSearch()

def gradio_search(query, price_filter, rating_filter, search_type):
    try:
        filters = None
        if price_filter and rating_filter:
            filters = (
                Filter.by_property("stock_status").equal("In stock") &
                Filter.by_property("price").less_or_equal(float(price_filter)) &
                Filter.by_property("rating").greater_or_equal(float(rating_filter))
            )
        elif price_filter:
            filters = (
                Filter.by_property("stock_status").equal("In stock") &
                Filter.by_property("price").less_or_equal(float(price_filter))
            )
        elif rating_filter:
            filters = (
                Filter.by_property("stock_status").equal("In stock") &
                (Filter.by_property("rating").greater_or_equal(float(rating_filter)) | Filter.by_property("rating").is_none(True))
            )
        else:
            filters = (
                Filter.by_property("stock_status").equal("In stock")
            )
        
        if search_type == "Hybrid Search":
            return "\n\n".join(search_engine.hybrid_search(query, filters=filters))
        elif search_type == "Keyword Search":
            return "\n\n".join(search_engine.keyword_search(query, filters=filters))
        else:
            return "Invalid search type."
    except Exception as e:
        logging.error(f"Error processing search: {e}")
        return "Error: Search failed."

iface = gr.Interface(
    fn=gradio_search,
    inputs=[
        gr.Textbox(label="Search Query"),
        gr.Number(label="Max Price (Optional)", value=None),
        gr.Number(label="Min Rating (Optional)", value=None),
        gr.Radio(["Hybrid Search", "Keyword Search"], label="Search Type")
    ],
    outputs="html",
    title="AI-Powered Product Search"
)

iface.launch()
