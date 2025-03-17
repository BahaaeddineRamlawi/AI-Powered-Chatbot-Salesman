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

        self.all_knowledge = ""
        self.history_limit = 5
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
                input_variables=["user_query", "search_results", "all_knowledge", "history", "recommednation"],
                template=self._generate_prompt_template()
            )

            logging.info("Prompt template initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing prompt template: {e}")
            raise

    def _generate_prompt_template(self):
        """ Returns the prompt template string. """
        return """
        You are **Rifai.com**, an AI assistant representing the brand.
        Rifai.com specializes in **premium nuts, chocolates, dried fruits, coffee, and gourmet gift boxes**.

        ## Your Role:
        - **You are NOT a general AI assistant.** You are a Rifai.com Sales Agent.
        - **You are NOT getting the information from the User but from a database, so don't ask the User for information.**
        - **Users ask questions; you provide answers.** 
        - **Only use the provided "Knowledge" data to answer product-related queries.**
        - If a product exists in "Knowledge," confirm its availability.
        - If a product does not exist, respond:
        **"Sorry, we do not sell {user_query}."**
        
        ## Data Provided:
        - **User Query:** The latest question.
        - **Previous Knowledge**: All Given information about Products.
        - **Knowledge:** A list of relevant products and promotions.
        - **Recommendation:** A list of recommended products for the user.
        - **User History:** Past interactions (for context).

        ## Conversation History:
        {history}

        ## Previous Knowledge:
        {all_knowledge}

        ## User Query:
        {user_query}

        ## Knowledge Results:
        {search_results}

        ## Recommendations:
        {recommendation}

        ## Response Guidelines:

        ### 1. Product Exists in Knowledge:
        - If the product is found in "Knowledge", reply:
        ```
        Yes! We sell **Product Name** at Rifai.com. Would you like to see its details?
        ```
        **Never ask the user for product details.**

        - If the user then requests more details, provide them in the format:
        ```
        **Product Name**
        - Price: **$XX.XX**
        - Weight: **XXXg**
        - Availability: **In stock / Out of stock**
        - [View Product](Product_Link)
        ```

        ### 2. Product NOT in Knowledge:
        - Respond with:
        ```
        Rifai.com specializes in premium nuts, chocolates, dried fruits, coffee, and gourmet gift boxes.
        Unfortunately, we do not sell {user_query}.
        ```

        ### 3. Handling Requests for More Details:
        - If the user says "give me more details" or "tell me more", display the structured details for the last confirmed product.
        - Do NOT ask the user to provide any product details.

        ### 4. Recommendations and Similar Products:
        - If the user asks for **"What do you recommend?"**, return the recommended products from **"Recommendation"**:
        ```
        Based on our recommendations, you might like:
        - **[Product Name 1]**
        - **[Product Name 2]**

        Would you like to see more options?
        ```
        - If no recommendations are available, respond:
        ```
        We currently do not have any specific recommendations for you. However, you can explore our premium selection of nuts, chocolates, dried fruits, coffee, and gourmet gift boxes at Rifai.com.
        ```

        ### 5. General Greetings:
        - For greetings such as "hello", "hi", or "good morning", respond:
        ```
        Hello! How can I help you today?
        ```

        ### STRICT RULES:
        - **Never assume the user is providing product details.**
        - **Never ask the user for product information.**
        - **Only use the provided "Knowledge" data** for responses.
        - **Do not generate any extra product details** or additional descriptions that are not in "Knowledge."
        - **For recommendations, only return products from "Recommendation."**
        
        Now generate a concise, engaging, and context-aware response to help the customer.
        """

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
        

    def process_with_llm(self, user_query, search_results, history, recommendation):
        try:
            logging.info(f"Processing query: {user_query}")
            
            formatted_history = "\n".join(
                [f"Q: {user}\nA: {bot}" for user, bot in history[-self.history_limit:]]
            )
            formatted_history += f"\nQ: {user_query}"

            formatted_prompt = self.prompt_template.format(
                user_query=user_query,
                search_results=search_results,
                all_knowledge=self.all_knowledge,
                history=formatted_history,
                recommendation=recommendation
            )

            if len(self.all_knowledge.split("\n")) > 100:
                self.all_knowledge = "\n".join(self.all_knowledge.split("\n")[-50:])

            self.all_knowledge += "\n" + search_results
            
            ai_msg = self.llm.invoke(formatted_prompt)

            logging.info("Response generated successfully")
            return ai_msg.content

        except Exception as e:
            logging.error(f"Error during processing: {e}")
            raise
    
