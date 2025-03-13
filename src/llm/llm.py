from langchain_openai import AzureChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from src.utils import logging, config

class LLMHandler:
    def __init__(self):
        self.llm_provider = config["llm"]["provider"]
        try:
            logging.info(f"Initializing LLMHandler with provider {self.llm_provider}")
            llm_mapping = {
                "openai": self._init_openai,
                "azure_openai": self._init_azure_openai,
                "gemini": self._init_gemini ,
                "mistral": self._init_mistral,
                "llama3-70b-8192": self._init_groq,
                "mixtral-8x7b-32768": self._init_groq,
                "gemma-7b-it": self._init_groq,
            }

            if self.llm_provider in llm_mapping:
                llm_mapping[self.llm_provider]()
            else:
                raise ValueError(f"Invalid LLM provider: {self.llm_provider}")

            logging.info(f"LLMProvider {self.llm_provider} initialized successfully")

        except Exception as e:
            logging.error(f"Error initializing LLMProvider: {e}")
            raise

        try:
            self.prompt_template = PromptTemplate(
                input_variables=["user_query", "search_results", "history"],
                template="""
                You are an AI-powered salesman assistant for **Rifai.com**, helping users find the best products based on their queries. 
                Rifai.com specializes in selling **premium nuts, chocolates, dried fruits, coffee, and gourmet gift boxes**.
                ## Your Primary Goal:
                - **Directly answer the user's query** using only the given product information.
                - **Prioritize the conversation history** to maintain continuity and context.
                - If product information is available, provide a relevant response with product details.
                - If no matching products exist, **do not make up information**. Instead, offer helpful alternatives.

                ## Provided Data:
                - **User History**: Previous interactions to maintain context and ensure a smooth conversation flow.
                - **User Query**: The latest request from the user.
                - **Available Products & Offers**: A list of relevant products and promotions.

                ## Conversation History:
                {history}

                ## User Query:
                {user_query}

                ## Available Products with Offers (Knowledge):
                {search_results}

                ### Instructions:
                1. **Conversation Continuity (Focus on History First):**
                - **Before answering, always check the conversation history.**
                - If the user has already asked a similar question, **avoid redundant responses**—instead, build upon previous answers.
                - If the user is referring to a past response (e.g., "Tell me more about that product"), ensure your answer aligns with the prior conversation.
                - If the user has previously expressed a preference (e.g., "I prefer dark chocolate"), **take that into account** when recommending products.

                2. **Product Selection Criteria:**
                - Initially, return only **2** products.
                - Ask the user if they would like to see more options.
                - If an exact match is unavailable, choose the closest matching product based on the user query.
                - Make sure the selected products are still relevant to the user's request.

                3. **If the user's question is related to products or offers:**
                - ONLY use the provided product list and offers.
                - Ensure that all products mentioned are only those present in the given list.
                - Mention Links for the provided products.
                - If there are active offers, **mention them explicitly before listing products**.
                - **Only include product images if the user explicitly requests them.** Otherwise, do not show images.

                4. **If the user asks for product details:**
                - Provide the requested details, including price, weight, rating, and availability.
                - If an offer is available for the product, mention it.
                - If the user asks about a specific product that is not in the provided list, inform them that it is not available.

                5. **If the user's question is a general greeting (e.g., "hello", "hi", "good morning"):**
                - Simply respond: **"Hello! How can I help you today?"**
                - Do not add extra information or unnecessary conversation.

                ### IMPORTANT:
                - **Never ignore previous interactions—use them to improve responses.**
                - **Never add extra products or offers that are not explicitly provided.**
                - Never generate imaginary product names, images, or details.
                - Always mention offers if there are any.
                - Prioritize clarity, accuracy, and helpfulness.

                Now generate a concise, engaging, and context-aware response.
                """
            )

            logging.info("Prompt template initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing prompt template: {e}")
            raise

    def stream(self,message):
        return self.llm.stream(message)

    def _init_openai(self):
        self.llm = ChatOpenAI(
            openai_api_key=config["openai"]["api_key"],
            temperature=config["llm"]["temperature"],
            model=config["openai"]["model"]
        )

    def _init_azure_openai(self):
        self.llm = AzureChatOpenAI(
            deployment_name=config["azure_openai"]["azure_deployment"],
            openai_api_version=config["azure_openai"]["api_version"],
            temperature=config["llm"]["temperature"],
            openai_api_key=config["azure_openai"]["api_key"],
            azure_endpoint=config["azure_openai"]["azure_endpoint"],
            max_retries=2
        )
    
    def _init_gemini(self):
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=config["gemini"]["api_key"],
            temperature=config["llm"]["temperature"],
            model=config["gemini"]["model"]
        )

    def _init_mistral(self):
        self.llm = ChatMistralAI(
            api_key=config["mistral"]["api_key"],
            model=config["mistral"]["model"],
            temperature=config["llm"]["temperature"],
            max_retries=2,
        )
    
    def _init_groq(self):
        self.llm = ChatGroq(
            model=self.llm_provider, 
            groq_api_key=config["groq"]["api_key"], 
            temperature=config["llm"]["temperature"],
            max_retries=2,
        )
        

    def process_with_llm(self, user_query, search_results, history):
        try:
            logging.info(f"Processing query: {user_query}")
            formatted_history = "\n".join([f"User: {user}\nAssistant: {bot}" for user, bot in history[-5:]])

            formatted_prompt = self.prompt_template.format(
                user_query=user_query,
                search_results=search_results,
                history=formatted_history
            )
            
            ai_msg = self.llm.invoke(formatted_prompt)

            logging.info("Response generated successfully")
            return ai_msg.content

        except Exception as e:
            logging.error(f"Error during processing: {e}")
            raise
    
