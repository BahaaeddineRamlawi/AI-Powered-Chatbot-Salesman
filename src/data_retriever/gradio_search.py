import gradio as gr
import logging
from weaviate.classes.query import Filter

class GradioSearchApp:
    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.interface = gr.Interface(
            fn=self.gradio_search,
            inputs=[
                gr.Textbox(label="Search Query"),
                gr.Number(label="Max Price (Optional)", value=None),
                gr.Number(label="Min Rating (Optional)", value=None),
                gr.Radio(["Hybrid Search", "Keyword Search"], label="Search Type")
            ],
            outputs="html",
            title="AI-Powered Product Search"
        )
    
    def gradio_search(self, query, price_filter, rating_filter, search_type):
        try:
            filters = Filter.by_property("stock_status").equal("In stock")
            
            if price_filter:
                filters &= Filter.by_property("price").less_or_equal(float(price_filter))
            if rating_filter:
                filters &= (Filter.by_property("rating").greater_or_equal(float(rating_filter)) | Filter.by_property("rating").is_none(True))
            
            if search_type == "Hybrid Search":
                return "\n\n".join(self.search_engine.hybrid_search(query, filters=filters))
            elif search_type == "Keyword Search":
                return "\n\n".join(self.search_engine.keyword_search(query, filters=filters))
            else:
                return "Invalid search type."
        except Exception as e:
            logging.error(f"Error processing search: {e}")
            return "Error: Search failed."
    
    def launch(self):
        try:
            self.interface.launch()
        finally:
            self.search_engine.close()
