from src.llm import ChatbotHandler
from src.utils import logging

if __name__ == "__main__":
    try:
        chatbot_handler = ChatbotHandler()
        chatbot_handler.launch_chatbot()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
