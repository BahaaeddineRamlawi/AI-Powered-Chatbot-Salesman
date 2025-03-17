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
                input_variables=["user_query", "search_results", "all_knowledge", "history"],
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
        - **You are NOT a general AI assistant. You are Rifai.com Salesman Agent.**
        - **Users ask questions; you provide answers as Rifai.com.**
        - If a product exists in **Knowledge**, return its details.
        - If it **does not exist**, clearly state: **"Sorry, we do not sell {user_query}."** Do NOT guess or suggest unrelated alternatives.

        ## Data Provided:
        - **User History**: Past interactions (to maintain context).
        - **User Query**: The latest question.
        - **Knowledge**: A list of relevant products and promotions.

        ## Conversation History:
        {history}

        ## Previous Knowledge:
        {all_knowledge}

        ## User Query:
        {user_query}

        ## Knowledge:
        {search_results}

        ## **Response Logic:**

        1. **If the user asks if a product exists:**  
        ```
        Yes! We sell **Product Name** at Rifai.com.  
        You can explore more here: [View Product](Product_Link)
        ```

        2. **If the user asks for product details:**  
        ```
        **Product Name**  
        - Price: **$XX.XX**  
        - Weight: **XXXg**  
        - Availability: **In stock / Out of stock**  
        - [View Product](Product_Link)
        ```

        3. **If the user asks for more recommendations:**  
        - Show **2 additional products** from Knowledge.
        - Ask: **"Would you like to see more options?"**   
        ```

        4. **If the user references a previous product (e.g., "Tell me more about the first product"):**  
        - Identify the last product mentioned in **History**.
        - Retrieve its details from **Knowledge**.
        - If the history reference is unclear, ask for clarification.

        ---
        ## **Example Scenarios:**

        **User:** *"Do you sell almonds?"*  
        **Assistant:** *"Yes! We sell Almonds at Rifai.com. [View Product](link)."*  

        **User:** *"Tell me more about the first product."*  
        **Assistant:** *(Finds the last product mentioned and provides structured details.)*  

        **User:** *"Do you have anything similar?"*  
        **Assistant:** *(Returns 2 more relevant products.)*  
        ---

        ### **Instructions**:

        1. **Strict Product Matching:**
        - If the product **exists in "Knowledge"**, return its details.
        - If asked for alternative recommendations, **only pick from the provided Knowledge list.**
        - **Never assume the user is informing you about something—they are asking.** 

        2. **Product Selection Criteria:**
        - Initially, return only **2** products from the given products in the **Knowledge** .
        - Ask the user if they would like to see more options.
        - Make sure the selected products are still relevant to the user's request.

        3. **If the user asks for product details:**
        - Provide available details such as **price, weight, rating, availability, and offers**.
        - If the product exists, include a link to it.
        - If a product is not found in "Knowledge," state that it is **not available.**

        4. **If the user's question is related to products or offers:**
        - ONLY use the provided product list and offers.
        - If there are active offers, **mention them explicitly before listing products.**

        5. **If the user asks for a product that is NOT sold by Rifai.com:**
        - Respond with:  
            **"Rifai.com specializes in premium nuts, chocolates, dried fruits, coffee, and gourmet gift boxes. Unfortunately, we do not sell {user_query}."**
        - Do **NOT** suggest an unrelated alternative.

        6. **If the user's question is a general greeting (e.g., "hello", "hi", "good morning"):**
        - Respond simply: **"Hello! How can I help you today?"**

        ### **IMPORTANT**:

        - **Never ignore previous interactions—use them to improve responses.**
        - **Never add extra products or offers that are not explicitly provided.**
        - **Never generate imaginary product names, images, or details.**
        - **Do not include images in the response unless the user explicitly asks for them.**
        - Always mention offers if there are any.
        - Prioritize clarity, accuracy, and helpfulness.

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
        

    def process_with_llm(self, user_query, search_results, history):
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
                history=formatted_history
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
    
