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
                input_variables=["user_query", "search_results", "history"],
                template=self._generate_prompt_template()
            )

            logging.info("Prompt template initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing prompt template: {e}")
            raise

    def _generate_prompt_template(self):
        """ Returns the prompt template string. """
        return """
        ## Context
        ### User History:
        {history}

        ### User Query:
        {user_query}

        ### Knowledge:
        {search_results}

        ## Response Guidelines:

        ### 1. Product Exists in Knowledge:
        - If the product is found in "Knowledge", then answer based on "Knowledge".

        ### 2. Product NOT in Knowledge:
        - Respond with:
        ```
        Rifai.com specializes in premium nuts, chocolates, dried fruits, coffee, and gourmet gift boxes.
        Unfortunately, we do not sell {user_query}.
        ```
        
        ### 3. Other Chat:
        - If the User greets (e.g., "Hello," "Hi," "Good morning", "How are you", ...), reply to him in a very briefly way in a half line and add that you will assist him as an assistant for Rifai.com.
        - If the User is chatting, reply to him with one word and add that you will assist him as an assistant for Rifai.com.
        
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
                history=formatted_history
            )
            
            ai_msg = self.llm.invoke(formatted_prompt)

            logging.info("Response generated successfully")
            return ai_msg.content

        except Exception as e:
            logging.error(f"Error during processing: {e}")
            raise
    
