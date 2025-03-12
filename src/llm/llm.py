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

                You will be given:  
                - **User History**: Previous interactions with the user to maintain context.  
                - **User Query**: The latest question or request from the user.  
                - **Available Products & Offers**: A list of relevant products and any available offers.  

                ## Conversation History:
                {history}

                ## User Query:
                {user_query}

                ## Available Products with Offers (Knowledge):
                {search_results}

                ### Instructions:
                1. **Product Selection Criteria:**
                - Always return up to **5** products.
                - If an exact match is unavailable, choose the closest matching product based on the user query.
                - Make sure the selected products are still relevant to the user's request.

                2. **If the user's question is related to products or offers:**
                - ONLY use the provided product list and offers.
                - Ensure that all products mentioned are only those present in the given list.
                - Always provide available offers and **explicitly mention that offers are available** before listing products.
                - Include product images where available (with a size of **200px max**).

                3. **If the user's question is a general greeting or unrelated to products:**
                - Respond naturally, as a human would.
                - For greetings like "hello", keep the response friendly and short without providing irrelevant information.

                4. **If the user asks about products not sold by Rifai.com:**
                - Inform them: **"Rifai.com specializes in premium nuts, chocolates, dried fruits, coffee, and gourmet gift boxes. We do not sell {user_query}."**
                - Suggest similar products if available.

                5. **If no products are found:**
                - Respond with: **"Sorry, no products match your query at the moment. Try adjusting your search criteria or checking related categories."**

                ### IMPORTANT:
                - Never add extra products or offers that are not explicitly provided.
                - Never generate imaginary product names, images, or details.
                - Always mention offers if there are any.
                - Prioritize clarity, accuracy, and helpfulness.

                Now generate a concise, engaging, and visually appealing response.
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
    
