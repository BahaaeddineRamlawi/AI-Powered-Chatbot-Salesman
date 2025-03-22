from src.llm import ChatbotHandler
from src.utils import logging
import atexit

def on_shutdown(chatbot_handler):
    """Shutdown function to properly close resources when the program ends."""
    chatbot_handler.shutdown()

if __name__ == "__main__":
    try:
        chatbot_handler = ChatbotHandler()
        chatbot_handler.launch_chatbot()
        atexit.register(on_shutdown, chatbot_handler)       
    except Exception as e:
        logging.error(f"Fatal error: {e}")
