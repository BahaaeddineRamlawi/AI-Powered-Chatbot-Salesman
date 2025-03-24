from langchain_openai import AzureChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os

from src.utils import logging, config

class LLMHandler:
    def __init__(self):
        self.llm_provider = config["llm"]["provider"]
        self.prompt_files = {
            "default": config["prompt_templates"]["main_prompt_template_path"],
            "greeting": config["prompt_templates"]["greeting_prompt_template_path"],
        }
        
        self._initialize_llm()
        
        logging.info(f"LLMProvider {self.llm_provider} initialized successfully")

    def _initialize_llm(self):
        """Initializes the correct LLM based on provider"""
        llm_mapping = {
            "openai": self._init_openai,
            "azure_openai": self._init_azure_openai,
            "gemini": self._init_gemini,
            "mistral": self._init_mistral,
            "llama3-70b-8192": self._init_groq,
            "mixtral-8x7b-32768": self._init_groq,
            "gemma-7b-it": self._init_groq,
        }

        if self.llm_provider in llm_mapping:
            llm_mapping[self.llm_provider]()
        else:
            raise ValueError(f"Invalid LLM provider: {self.llm_provider}")


    def _load_prompt_template(self, template_name):
        """Loads a prompt template and wraps it with PromptTemplate"""
        if template_name not in self.prompt_files:
            raise ValueError(f"Invalid template name: {template_name}")

        prompt_path = self.prompt_files[template_name]

        if not os.path.exists(prompt_path):
            logging.error(f"Prompt file {prompt_path} not found.")
            raise FileNotFoundError(f"Prompt file {prompt_path} not found.")

        try:
            with open(prompt_path, "r", encoding="utf-8") as file:
                logging.info(f"Successfully loaded prompt template from {prompt_path}")

                if template_name == "greeting":
                    input_vars = ["store_name"]
                else:
                    input_vars = ["user_query", "search_results", "history"]

                return PromptTemplate(template=file.read(), input_variables=input_vars)
        except Exception as e:
            logging.error(f"Error reading prompt template: {e}")
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
        

    def process_with_llm(self, user_query, search_results, history, intent, template_name="default"):
        try:
            logging.info(f"Processing query: {user_query}")

            prompt_template = self._load_prompt_template(template_name)
            if template_name == "greeting":
                formatted_prompt = prompt_template.format(
                    store_name=config["store"]["name"],
                )
            else:
                formatted_prompt = prompt_template.format(
                    user_query=user_query,
                    search_results=search_results,
                    history=history,
                    intent=intent
                )
            ai_msg = self.llm.invoke(formatted_prompt)

            logging.info("Response generated successfully")
            return ai_msg.content

        except Exception as e:
            logging.error(f"Error during processing: {e}")
            raise
    
