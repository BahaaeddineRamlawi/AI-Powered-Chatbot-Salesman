import weaviate
import logging
import gradio as gr
from sentence_transformers import SentenceTransformer
from datetime import datetime
from weaviate.classes.query import Filter
import sqlite3


logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    filename=f"./logs/app_log_{datetime.now().strftime('%Y-%m-%d')}.log",
    filemode='a'
)

class WeaviateSearch:
    def __init__(self, collection_name="Product", model_name="all-MiniLM-L6-v2", db_path="offers_database.db"):
        """Initialize Weaviate connection, model, and SQLite database."""
        self.client = weaviate.connect_to_local()
        self.collection = self.client.collections.get(collection_name)
        self.model = SentenceTransformer(model_name)
        self.db_path = db_path
        logging.info(f"Connected to Weaviate and loaded model: {model_name}")

    def find_offers_by_product(self, product_id):
        """Find offers that contain a given product ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT * FROM offers
            WHERE ? IN (SELECT value FROM json_each(product_list))
        """
        cursor.execute(query, (str(product_id),))  # Convert product_id to string

        results = cursor.fetchall()

        conn.close()
        return results

    def hybrid_search(self, query, alpha=0.5, limit=5, filters=None):
        """Perform hybrid search using keyword & vector similarity."""
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

    def keyword_search(self, query, limit=3, filters=None):
        """Perform BM25 keyword search."""
        response = self.collection.query.bm25(
            query=query, limit=limit, filters=filters
        )
        return self._format_results(response)

    def _format_results(self, response):
        """Format search results into a structured response."""
        results = []
        for index, obj in enumerate(response.objects, 1):
            product_id = obj.properties.get("product_id", "Unknown")
            title = obj.properties.get("title", "No Title")
            # description = obj.properties.get("description", "No Description").replace("\n", "<br>")
            categories = obj.properties.get("categories", "No Category")
            image_url = obj.properties.get("image", "none")
            price = obj.properties.get("price", "N/A")
            rating = obj.properties.get("rating", "N/A")
            
            # Find offers for the product
            offers = self.find_offers_by_product(product_id)
            offers_html = ""
            if offers:
                offers_html = "<h3>Related Offers:</h3><ul>"
                for offer in offers:
                    offers_html += f"<li>{offer[1]} - {offer[2]}</li>"
                offers_html += "</ul>"

            result_html = f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px;">
                <h1>Item {index}</h1>
                <img src="{image_url}" alt="Product Image" style="width:200px; height:auto; border-radius:10px;">
                <h2>{title}</h2>
                <p><strong>Categories:</strong> {categories}</p>
                <p><strong>Price:</strong> ${price}</p>
                <p><strong>Rating:</strong> {rating} ⭐</p>
                {offers_html}
            </div>
            """
            # <p>{description}</p>
            
            results.append(result_html)
        
        
        logging.info(f"Found {len(results)} results")
        return results if results else ["No matching items found."]

    def close(self):
        """Close Weaviate connection."""
        self.client.close()
        logging.info("Weaviate connection closed.")

search_engine = WeaviateSearch()

def gradio_search(query, price_filter, rating_filter, search_type):
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

iface = gr.Interface(
    fn=gradio_search,
    inputs=[
        gr.Textbox(label="Search Query"),
        gr.Number(label="Max Price (Optional)", value=None),
        gr.Number(label="Min Rating (Optional)", value=None),
        gr.Radio(["Hybrid Search", "Keyword Search"], label="Search Type")
    ],
    outputs="markdown",
    title="AI-Powered Product Search"
)

iface.launch()
