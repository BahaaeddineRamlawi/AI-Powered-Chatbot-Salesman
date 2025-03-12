import gradio as gr

from .llm import LLMHandler
from src.data_retriever import WeaviateHandler, RecommendationHandler
from src.utils import logging


class ChatbotHandler:
    def __init__(self):
        self.llmhandler = LLMHandler()
        self.search_engine = WeaviateHandler()
        self.recommendation_engine = RecommendationHandler()

    def stream_response(self, message, history):
        """
        Handle streaming responses to the chat interface.
        If knowledge is not found or if an error occurs, it logs the event.
        """
        try:
            knowledge,product_ids = self.search_engine.hybrid_search(message)
            if not product_ids:
                recommendation_items = "No product IDs available for recommendation."
            else:
                recommendation_items = self.recommendation_engine.hybrid_recommendation(2001, product_ids[0])
            
            print(recommendation_items)
            logging.info(f"Received message: {message}")

            if message is not None:
                partial_message = ""

                rag_prompt = self.llmhandler.process_with_llm(
                    message, knowledge, history
                )

                for response in self.llmhandler.stream(rag_prompt):
                    partial_message += response.content
                    yield partial_message
        except Exception as e:
            logging.error(f"Error during message processing: {e}")
            yield "Sorry, there was an error processing your request."

    def launch_chatbot(self):
        """Launch the Gradio Chatbot Interface"""
        try:
            chatbot = gr.ChatInterface(
                self.stream_response, 
                textbox=gr.Textbox(
                    placeholder="Send to the LLM...",
                    container=False,
                    autoscroll=True,
                    scale=7
                ),
            )
            logging.info("Launching Gradio chatbot.")
            chatbot.launch()
        except Exception as e:
            logging.error(f"Error launching chatbot: {e}")
            raise