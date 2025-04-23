import gradio as gr

from .llm import LLMHandler
from src.data_retriever import RecommendationHandler, QueryInfoExtractor, OffersDatabase, MarketingStrategies
from src.utils import logging
import random

MAX_QUERY_LENGTH = 300

class ChatbotHandler:
    def __init__(self, weaviate_handler):
        try:
            self.search_engine = weaviate_handler
            self.llmhandler = LLMHandler()
            self.filter_extractor = QueryInfoExtractor()
            self.offer_db = OffersDatabase()
            self.recommendation_engine = RecommendationHandler()
            self.strategies = MarketingStrategies(self.search_engine, self.recommendation_engine)
            self.history_limit = 8
            self.max_history_check = 2
            self.product_query_counter = 4
            self.question_to_ask = ""
            self.up_sell = False
            self.cross_sell = False
            self.cross_sell_percentage = 1
            self.up_sell_percentage = 1

        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            self.search_engine.close()
            raise


    def stream_response(self, query, history, user_id=2001):
        """
        Handle streaming responses to the chat interface.
        If knowledge is not found or if an error occurs, it logs the event.
        """
        try:
            if query is None or len(query.strip()) == 0:
                yield "Please enter a valid query."
                return
            
            if len(query) > MAX_QUERY_LENGTH:
                yield f"Your message is too long. Please keep it under {MAX_QUERY_LENGTH} characters."
                return
            
            knowledge, intent, features = self._get_knowledge(query=query, history=history,user_id=user_id)
            logging.info(f"Received query: {query}")

            if self.product_query_counter <= 5 and intent in {"ask_for_product", "ask_without_product"}:
                self.product_query_counter += 1
                if self.product_query_counter == 3:
                    self.question_to_ask = """\nAsk the user if they're interested in current deals - for example, with a question like: '**Would you like to check out our current offers?**'"""
                    
                elif self.product_query_counter == 4:
                    self.question_to_ask = """\nAsk the user if they'd like a recommendation based on their past ratings - for example, with a question like: '**Would you like to see personalized recommendations based on what you've liked before?**'"""
                
                elif self.product_query_counter == 5:
                    self.question_to_ask = """\nAsk the user this question at the end — '**Would you prefer a more premium option?**' or a similar question."""

                elif self.product_query_counter == 6:
                    self.question_to_ask = """\nAsk the user this question at the end — '**Are you thinking of pairing this with anything?**'"""
                       
                logging.info(f"Question To Ask: {self.question_to_ask != ''}")
            
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
                    query, knowledge, formatted_history, intent, features, self.question_to_ask
                )
                for response in self.llmhandler.stream(rag_prompt):
                    partial_message += response.content
                    yield partial_message
            self.question_to_ask = ""

        except Exception as e:
            logging.error(f"Error during message processing: {e}")
            yield "Sorry, there was an error processing your request."
    

    def _get_recommendation_data(self, intent, user_id=None):
        """Get recommendations based on the intent"""
        if intent == "ask_for_recommendation" and user_id:
            return self.recommendation_engine.get_user_based_recommendations(user_id=int(user_id))
        
        else:
            return "Insufficient data to generate recommendations."
    

    def _get_cross_selling_data(self, base_product):
        try:
            return self.strategies.get_cross_sell_recommendations(base_product)
        
        except Exception as e:
            logging.error(f"Error fetching cross-selling data: {e}")
            return ""


    def _get_up_selling_data(self, base_product):
        try:
            return self.strategies.get_up_sell_recommendations(base_product)
        
        except Exception as e:
            logging.error(f"Error fetching up-selling data: {e}")
            return ""


    def _get_knowledge(self, query, history, user_id):
        try:
            knowledge = ""
            combined_query = query
            filters, intent, features, up_sell, cross_sell = self.filter_extractor.extract_info_from_query(query, history)
            if up_sell is not None:
                logging.info(f"Up_Selling: {up_sell}")
                self.up_sell = up_sell
                self.up_sell_percentage = 1

            if cross_sell is not None:
                logging.info(f"Cross_Selling: {cross_sell}")
                self.cross_sell = cross_sell
                self.cross_sell_percentage = 1

            final_combined_query = query
            features_string = ", ".join(features)

            if (intent == "follow_up"):
                logging.info("Intent is 'ask_without_product'. Combining with previous queries.")
                combined_queries = [query]
                user_queries = [entry["content"] for entry in reversed(history) if entry["role"] == "user"]
                for past_query in user_queries[:self.max_history_check]:
                    combined_queries.append(past_query)
                    combined_query = ", ".join(reversed(combined_queries))
                    _, past_intent, features, _, _ = self.filter_extractor.extract_info_from_query(past_query, history)
                    logging.info(f"After appending, combined query: {combined_query}")
                    if past_intent == "ask_for_product":
                        break
                final_combined_query = ", ".join(reversed(combined_queries))

            elif intent == "ask_for_unrelated_product":
                logging.info("Intent is 'ask_for_unrelated_product'")
                return "We don't have these products", intent, "No Features"

            elif intent == "ask_for_offers":
                logging.info("Intent is 'ask_for_offers'")
                knowledge = self.offer_db.get_offers()
                return knowledge, intent, features_string

            elif intent in ["gibberish", "greeting", "feedback", "other"]:
                logging.info(f"Intent is '{intent}'")
                return "", intent, "No Features"

            elif intent == "ask_for_recommendation":
                logging.info("Intent is 'ask_for_recommendation'")
                knowledge = self._get_recommendation_data(intent, user_id=user_id)
                return knowledge, intent, features_string

            knowledge, base_product = self.search_engine.hybrid_search(
                query=final_combined_query + ". " + features_string, filters=filters
            )

            if not base_product:
                logging.info("No base product found.")
            else:
                logging.info("Base product was found.")
                if self.up_sell and random.random() <= self.up_sell_percentage:
                    logging.info("Getting Up Selling Products")
                    up_sell_knowledge = "**Up Selling Products**:\n" + self._get_up_selling_data(base_product)
                    if "No Product Available or Requested" not in up_sell_knowledge:
                        knowledge = up_sell_knowledge
                    self.up_sell_percentage = 0.35

                if self.cross_sell and random.random() <= self.cross_sell_percentage:
                    logging.info("Getting Cross Selling Products")
                    cross_sell_knowledge = "**Cross Selling Products**: (People often pair the previous product with these Products)\n"
                    cross_sell_knowledge += self._get_cross_selling_data(base_product)
                    knowledge = cross_sell_knowledge
                    self.cross_sell_percentage = 0.35
            print(knowledge)
            return knowledge, intent, features_string
        except Exception as e:
            logging.error(f"Error retrieving knowledge: {e}")
            return "", "error", "No Features"


    def launch_chatbot(self):
        """Launch the Gradio Chatbot Interface"""
        try:
            user_id_input = gr.Textbox(
                label="User ID (Optional)", 
                placeholder="Default is 2001", 
                value="2001",
                lines=1
            )

            chatbot = gr.ChatInterface(
                fn=self.stream_response, 
                textbox=gr.Textbox(
                    placeholder="Send to the LLM...",
                    container=False,
                    autoscroll=True,
                    scale=7
                ),
                additional_inputs=[user_id_input],
                autofocus=True,
                type="messages"
            )
            logging.info("Launching Gradio chatbot.")
            chatbot.launch()
        except Exception as e:
            logging.error(f"Error launching chatbot: {e}")
            raise


    def shutdown(self):
        try:
            logging.info("Shutting down the chatbot and closing Weaviate connection.")
            self.search_engine.close()
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")