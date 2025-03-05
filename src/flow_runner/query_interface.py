from src.data_retriever import WeaviateHandler, GradioChatbotApp

if __name__ == "__main__":
    weaviate_handler = WeaviateHandler()
    app = GradioChatbotApp(weaviate_handler)
    app.launch()
