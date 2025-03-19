import gradio as gr

from .llm import LLMHandler
from src.data_retriever import WeaviateHandler, RecommendationHandler
from src.utils import logging


class ChatbotHandler:
    def __init__(self):
        self.search_engine = WeaviateHandler()
        # self.recommendation_engine = RecommendationHandler()
        self.llmhandler = LLMHandler()
        # self.recommended_products = self.recommendation_engine.get_hybrid_recommendations(user_id=2001)
        
        # self.recommendation_str = "Top 5 hybrid recommendations for you:\n"
        # for idx, (product, _) in enumerate(self.recommended_products, 1):
        #     self.recommendation_str += f"{idx}. {product}\n"

    def stream_response(self, message, history):
        """
        Handle streaming responses to the chat interface.
        If knowledge is not found or if an error occurs, it logs the event.
        """
        try:
            knowledge = self.search_engine.hybrid_search(message)
            logging.info(f"Received message: {message}")
            # print(self.recommendation_str)

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