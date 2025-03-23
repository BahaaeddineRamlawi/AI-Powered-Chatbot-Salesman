import gradio as gr

from .llm import LLMHandler
from src.data_retriever import WeaviateHandler, RecommendationHandler, QueryInfoExtractor
from src.utils import logging

class ChatbotHandler:
    def __init__(self):
        try:
            self.search_engine = WeaviateHandler()

            # self.recommendation_engine = RecommendationHandler()

            self.llmhandler = LLMHandler()
            self.filter_extractor = QueryInfoExtractor()
            self.history_limit = 5

            # self.recommended_products = self.recommendation_engine.get_hybrid_recommendations(user_id=2001)
            
            # self.recommendation_str = "Top 5 hybrid recommendations for you:\n"
            # for idx, (product, _) in enumerate(self.recommended_products, 1):
            #     self.recommendation_str += f"{idx}. {product}\n"

        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            self.search_engine.close()
            raise


    def stream_response(self, query, history):
        """
        Handle streaming responses to the chat interface.
        If knowledge is not found or if an error occurs, it logs the event.
        """
        try:
            knowledge = self._get_weaviate_data(query=query, history=history)
            logging.info(f"Received query: {query}")
            # print(self.recommendation_str)
            
            formatted_history = ""
            for i in range(0, len(history) - 1, 2):
                if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
                    formatted_history += f"Human: {history[i]['content']}\nMe: {history[i + 1]['content']}\n"
            formatted_history += f"\nHuman: {query}"

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
    

    def _get_weaviate_data(self, query, history):
        """Retrieve data from Weaviate based on query and intent, considering user history."""
        max_history_check = 3 
        combined_query = query
        filters, intent = self.filter_extractor.extract_info_from_query(query)
        final_combined_query = query

        if intent == "ask_for_details_without_product":
            logging.info("Intent is 'ask_for_details_without_product'. Combining with previous queries.")

            combined_queries = [query]

            user_queries = [entry["content"] for entry in reversed(history) if entry["role"] == "user"]

            for past_query in user_queries[:max_history_check]:

                combined_queries.append(past_query)

                combined_query = ", ".join(reversed(combined_queries))

                _, combined_intent = self.filter_extractor.extract_info_from_query(combined_query)
                logging.info(f"After appending, combined query: {combined_query} | {combined_intent}")
               
                if combined_intent != "ask_for_details_without_product":
                    break

            final_combined_query = ", ".join(reversed(combined_queries))
            logging.info(f"Final combined query: {final_combined_query}")

        knowledge = self.search_engine.hybrid_search(query=final_combined_query, filters=filters)

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


    def shutdown(self):
        """Cleanup and shutdown the Weaviate connection gracefully."""
        logging.info("Shutting down the chatbot and closing Weaviate connection.")
        self.search_engine.close()