import gradio as gr

from .llm import LLMHandler
from src.data_retriever import WeaviateHandler, RecommendationHandler, QueryFilterExtractor
from src.utils import logging

class ChatbotHandler:
    def __init__(self):
        self.search_engine = WeaviateHandler()
        # self.recommendation_engine = RecommendationHandler()
        self.llmhandler = LLMHandler()
        self.filter_extractor = QueryFilterExtractor()
        self.history_limit = 5
        # self.recommended_products = self.recommendation_engine.get_hybrid_recommendations(user_id=2001)
        
        # self.recommendation_str = "Top 5 hybrid recommendations for you:\n"
        # for idx, (product, _) in enumerate(self.recommended_products, 1):
        #     self.recommendation_str += f"{idx}. {product}\n"


    def stream_response(self, query, history):
        """
        Handle streaming responses to the chat interface.
        If knowledge is not found or if an error occurs, it logs the event.
        """
        try:
            knowledge = self._get_weaviate_data(query)
            logging.info(f"Received query: {query}")
            # print(self.recommendation_str)
            
            formatted_history = ""
            for i in range(0, len(history) - 1, 2):
                if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
                    formatted_history += f"Q: {history[i]['content']}\nA: {history[i + 1]['content']}\n"
            formatted_history += f"\nQ: {query}"

            if query is not None:
                partial_message = ""

                rag_prompt = self.llmhandler.process_with_llm(
                    query, knowledge, formatted_history
                )

                for response in self.llmhandler.stream(rag_prompt):
                    partial_message += response.content
                    yield partial_message
        except Exception as e:
            logging.error(f"Error during message processing: {e}")
            yield "Sorry, there was an error processing your request."
    

    def _get_weaviate_data(self, query):
        filters = self.filter_extractor.extract_filters_from_query(query)
        knowledge = self.search_engine.hybrid_search(query=query, filters=filters)
        return knowledge


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
                type="messages"
            )
            logging.info("Launching Gradio chatbot.")
            chatbot.launch()
        except Exception as e:
            logging.error(f"Error launching chatbot: {e}")
            raise