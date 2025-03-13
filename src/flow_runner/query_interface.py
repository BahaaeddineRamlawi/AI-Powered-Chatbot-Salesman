from src.data_retriever import WeaviateHandler, GradioChatbotApp
from src.utils import logging

if __name__ == "__main__":
    try:
        weaviate_handler = WeaviateHandler()
        logging.info("WeaviateHandler initialized successfully.")
        
        app = GradioChatbotApp(weaviate_handler)
        logging.info("GradioChatbotApp initialized successfully.")
        
        app.launch()
        logging.info("Gradio Chatbot app launched successfully.")
    
    except Exception as e:
        logging.error(f"Error during app launch: {e}")
