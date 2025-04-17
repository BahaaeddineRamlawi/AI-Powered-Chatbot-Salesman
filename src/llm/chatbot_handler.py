import gradio as gr

from .llm import LLMHandler
from src.data_retriever import WeaviateHandler, RecommendationHandler, QueryInfoExtractor, OffersDatabase
from src.utils import logging

class ChatbotHandler:
    def __init__(self):
        try:
            self.search_engine = WeaviateHandler()

            self.llmhandler = LLMHandler()
            self.filter_extractor = QueryInfoExtractor()
            self.offer_db = OffersDatabase()
            self.recommendation_engine = RecommendationHandler()
            self.history_limit = 8
            self.max_history_check = 3
            self.product_query_counter = 0
            self.offer_suggestion_enabled = False
            self.similarity_suggestion_enabled = False

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
            knowledge, intent, features = self._get_knowledge(query=query, history=history,user_id=2001)
            logging.info(f"Received query: {query}")

            
            product_intents = {"ask_for_product", "ask_for_recommendation", "ask_without_product"}

            if intent in product_intents and self.product_query_counter <= 5:
                self.product_query_counter += 1
                if self.product_query_counter == 3:
                    self.offer_suggestion_enabled = True
                elif self.product_query_counter == 5:
                    self.similarity_suggestion_enabled = True
                else:
                    self.offer_suggestion_enabled = False
                    self.similarity_suggestion_enabled = False
                    
            
            formatted_history = ""

            recent_history = history[-self.history_limit:] if len(history) > self.history_limit else history
            for i in range(0, len(recent_history) - 1, 2):
                if recent_history[i]["role"] == "user" and recent_history[i + 1]["role"] == "assistant":
                    formatted_history += (
                        f"User: {recent_history[i]['content']}\n"
                        f"Assistant: {recent_history[i + 1]['content']}\n"
                    )

            if query is not None:
                partial_message = ""
                rag_prompt = self.llmhandler.process_with_llm(
                    query, knowledge, formatted_history, intent, features, self.offer_suggestion_enabled, self.similarity_suggestion_enabled
                )

                for response in self.llmhandler.stream(rag_prompt):
                    partial_message += response.content
                    yield partial_message
        except Exception as e:
            logging.error(f"Error during message processing: {e}")
            yield "Sorry, there was an error processing your request."
    

    def _get_recommendation_data(self, intent, user_id=None):
        """Get recommendations based on the intent"""
        if intent == "ask_for_recommendation" and user_id:
            return self.recommendation_engine.get_user_based_recommendations(user_id=int(user_id))
        else:
            return "Insufficient data to generate recommendations."          


    def _get_knowledge(self, query, history,user_id):
        """Retrieve knowledge baseed on query and intent."""
        knowledge = ""
        combined_query = query
        filters, intent, features, categories = self.filter_extractor.extract_info_from_query(query, history)
        final_combined_query = query
    
        if (intent == "ask_without_product" and features == [] and categories == []) or (intent == "follow_up"):
            logging.info("Intent is 'ask_without_product'. Combining with previous queries.")

            combined_queries = [query]
            user_queries = [entry["content"] for entry in reversed(history) if entry["role"] == "user"]

            for past_query in user_queries[:self.max_history_check]:
                combined_queries.append(past_query)
                combined_query = ", ".join(reversed(combined_queries))
                _, past_intent, features, categories = self.filter_extractor.extract_info_from_query(past_query, history)
                logging.info(f"After appending, combined query: {combined_query}")
               
                if past_intent == "ask_for_product":
                    break

            final_combined_query = ", ".join(reversed(combined_queries))
            logging.info(f"Final combined query: {final_combined_query}")

        elif intent == "ask_for_unrelated_product":
            logging.info("Intent is 'ask_for_unrelated_product'")
            return "We don't have this products", intent, []
        
        elif intent == "ask_for_offers":
            logging.info("Intent is 'ask_for_offers'")
            knowledge =  self.offer_db.get_offers()
            return knowledge, intent, features
        
        elif intent in ["gibberish", "greeting", "feedback"]:
            logging.info(f"Intent is '{intent}'")
            return "", intent, []
        
        elif intent == "ask_for_recommendation":
            logging.info("Intent is 'ask_for_recommendation'")
            knowledge = self._get_recommendation_data(intent, user_id=user_id)
            return knowledge, intent, features
        
        features_string = ", ".join(features)
        knowledge, first_product = self.search_engine.hybrid_search(query=final_combined_query + ". " + features_string, filters=filters)
        # if first_product:
        #     product_id = first_product["product_id"]
        # if intent == "ask_for_similar_items":
        #     logging.info("Intent is 'ask_for_similar_items'")
        #     knowledge = self._get_recommendation_data(intent, product_id=product_id)
        #     return knowledge, intent, features
        return knowledge, intent, features


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
                autofocus=True,
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