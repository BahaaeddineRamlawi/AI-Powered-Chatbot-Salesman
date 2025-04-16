from src.llm import ChatbotHandler
from src.utils import logging
import atexit

should_update_data = False

def on_shutdown(chatbot_handler):
    """Shutdown function to properly close resources when the program ends."""
    chatbot_handler.shutdown()


def update_and_create_vector_db():
    logging.info("Updating data and creating vector database...")
    pass


def run_chatbot():
    logging.info("Starting chatbot...")
    try:
        chatbot_handler = ChatbotHandler()
        chatbot_handler.launch_chatbot()
        atexit.register(on_shutdown, chatbot_handler)       
    except Exception as e:
        logging.error(f"Fatal error: {e}")


def main():
    global should_update_data

    if should_update_data:
        update_and_create_vector_db()
    else:
        logging.info("Skipping update...")
    run_chatbot()


if __name__ == "__main__":
    main()
