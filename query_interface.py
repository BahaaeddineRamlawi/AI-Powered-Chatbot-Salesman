import weaviate
import logging
import gradio as gr
from sentence_transformers import SentenceTransformer
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    filename=f"./logs/app_log_{datetime.now().strftime('%Y-%m-%d')}.log",
    filemode='a'
)

class WeaviateHybridSearch:
    def __init__(self, collection_name="Product", model_name="all-MiniLM-L6-v2"):
        """Initialize Weaviate connection, model, and collection."""
        self.client = weaviate.connect_to_local()
        self.collection = self.client.collections.get(collection_name)
        self.model = SentenceTransformer(model_name)
        logging.info(f"Connected to Weaviate and loaded model: {model_name}")

    def hybrid_search(self, query, alpha=0.5, limit=5):
        """Perform hybrid search in Weaviate using text & vector search."""
        logging.info(f"Performing search for query: {query}")

        # Generate query embedding
        query_embedding = self.model.encode(query).tolist()

        # Perform hybrid search
        response = self.collection.query.hybrid(
            query=query,
            vector=query_embedding,
            alpha=alpha,
            limit=limit,
            target_vector="info_vector"
        )

        results = []
        for index, obj in enumerate(response.objects, 1):
            title = obj.properties.get("title", "No Title")
            description = obj.properties.get("description", "No Description")
            description = description.replace("\n", "<br>")
            categories = obj.properties.get("categories", "No Category")
            image_url = obj.properties.get("image", "none")

            result_html = f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px;">
                <h1>Item {index}</h1>
                <img src="{image_url}" alt="Product Image" style="width:200px; height:auto; border-radius:10px;">
                <h2>{title}</h2>
                <p>{description}</p>
                <p><strong>Categories:</strong> {categories}</p>
            </div>
            """

            results.append(result_html)

        logging.info(f"Found {len(results)} results")
        return results if results else ["No matching items found."]

    def close(self):
        """Close Weaviate connection."""
        self.client.close()
        logging.info("Weaviate connection closed.")

search_engine = WeaviateHybridSearch()

def gradio_search(query):
    return "\n\n".join(search_engine.hybrid_search(query))

# Launch Gradio app
iface = gr.Interface(fn=gradio_search, inputs="text", outputs="markdown", title="AI-Powered Product Search")
iface.launch()
