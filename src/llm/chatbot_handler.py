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
            self.history_limit = 8
            self.max_history_check = 3 

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
                    query, knowledge, formatted_history, intent, features
                )

                for response in self.llmhandler.stream(rag_prompt):
                    partial_message += response.content
                    yield partial_message
        except Exception as e:
            logging.error(f"Error during message processing: {e}")
            yield "Sorry, there was an error processing your request."
    

    def _get_recommendation_data(self, user_id):
        self.recommendation_engine = RecommendationHandler()
        recommended_products = self.recommendation_engine.get_user_based_recommendations(user_id=user_id)
            
        recommendation_str = ""
        for product in recommended_products:
            recommendation_str += f"Title: {product['title']}\n"
            recommendation_str += f"Description: {product['description']}\n"
            recommendation_str += f"Price: {product['price']}\n"
            recommendation_str += f"Weight: {product['weight']}\n"
            recommendation_str += f"Link: {product['link']}\n"
            recommendation_str += f"Image: {product['image']}\n\n"            

        return recommendation_str
    

    def _get_knowledge(self, query, history,user_id):
        """Retrieve knowledge baseed on query and intent."""
        knowledge = ""
        combined_query = query
        filters, intent, features, categories = self.filter_extractor.extract_info_from_query(query, history)
        final_combined_query = query
        # rerank_feature = ""

        # if 'cheap' in features:
        #     rerank_feature = 'cheap'
        # elif 'expensive' in features:
        #     rerank_feature = 'expensive'
    
        if (intent == "ask_without_product" and features == [] and categories == []):
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
        
        elif intent == "gibberish" or intent == "greeting":
            logging.info(f"Intent is '{intent}'")
            return "", intent, []
        
        elif intent == "ask_for_recommendation":
            logging.info("Intent is 'ask_for_recommendation'")
            knowledge = self._get_recommendation_data(int(user_id))
            return knowledge, intent, features
        
        features_string = ", ".join(features)
        knowledge = self.search_engine.hybrid_search(query=final_combined_query + ". " + features_string, filters=filters)
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