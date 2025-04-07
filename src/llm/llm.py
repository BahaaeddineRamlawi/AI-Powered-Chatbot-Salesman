from langchain_openai import AzureChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os
import requests
import json

from src.utils import logging, config

class LLMHandler:
    def __init__(self):
        self.llm_provider = config["llm"]["provider"]
        self.prompt_files = {
            "default": config["prompt_templates"]["main_prompt_template_path"],
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


    def _load_prompt_template(self):
        """Loads a prompt template and wraps it with PromptTemplate"""
        prompt_path = config["prompt_templates"]["main_prompt_template_path"]

        if not os.path.exists(prompt_path):
            logging.error(f"Prompt file {prompt_path} not found.")
            raise FileNotFoundError(f"Prompt file {prompt_path} not found.")

        try:
            with open(prompt_path, "r", encoding="utf-8") as file:
                logging.info(f"Successfully loaded prompt template from {prompt_path}")
                return PromptTemplate(template=file.read(), input_variables=["user_query", "search_results", "history"])
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
        

    def process_with_llm(self, user_query, search_results, history, intent, features):
        try:
            logging.info(f"Processing query: {user_query}")

            prompt_template = self._load_prompt_template()
            formatted_prompt = prompt_template.format(
                user_query=user_query,
                search_results=search_results,
                history=history,
                intent=intent,
                features=features
            )

            logging.info("Response generated successfully")
            return formatted_prompt

        except Exception as e:
            logging.error(f"Error during processing: {e}")
            raise
    
