from ..data_retriever.weaviate import WeaviateHandler
from ..data_retriever.gradio_search import GradioSearchApp

if __name__ == "__main__":
    weaviate_handler = WeaviateHandler()
    app = GradioSearchApp(weaviate_handler)
    app.launch()