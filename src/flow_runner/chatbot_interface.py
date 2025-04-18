from src.llm import ChatbotHandler
from src.data_retriever import WeaviateHandler
from src.utils import logging
import atexit

def on_shutdown(chatbot_handler):
    """Shutdown function to properly close resources when the program ends."""
    chatbot_handler.shutdown()

if __name__ == "__main__":
    try:
        weaviate_handler = WeaviateHandler()
        chatbot_handler = ChatbotHandler(weaviate_handler)
        chatbot_handler.launch_chatbot()
        atexit.register(on_shutdown, chatbot_handler)       
    except Exception as e:
        logging.error(f"Fatal error: {e}")
