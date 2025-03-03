from src.data_retriever import WeaviateHandler, GradioSearchApp

if __name__ == "__main__":
    weaviate_handler = WeaviateHandler()
    app = GradioSearchApp(weaviate_handler)
    app.launch()
