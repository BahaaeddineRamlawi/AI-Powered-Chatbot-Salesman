import json
from langchain_openai import AzureChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from weaviate.classes.query import Filter
import os

from src.utils import logging, config


class QueryInfoExtractor:
    """Class to extract filters from product search queries using LLM."""
    
    def __init__(self):
        """Initialize the filter extractor with necessary configurations."""
        self.llm_provider = config["llm"]["provider"]
        self.prompt_file_path = config["prompt_templates"]["filter_and_intent_prompt_template_path"]
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

        self.prompt_template = self.load_template_from_file()
        if self.prompt_template:
            logging.info("Template loaded successfully")
        
        
        self.prompt = PromptTemplate(input_variables=["query"], template=self.prompt_template)
        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)


    def load_template_from_file(self):
        if not os.path.exists(self.prompt_file_path):
            logging.error(f"Prompt file {self.prompt_file_path} not found.")
            raise FileNotFoundError(f"Prompt file {self.prompt_file_path} not found.")
        
        try:
            with open(self.prompt_file_path, "r") as file:
                template = file.read()
            logging.info(f"Successfully loaded prompt template from {self.prompt_file_path}")
            return template
        except Exception as e:
            logging.error(f"Error loading template from file: {e}")
            return None
    
    
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
        

    def _convert_to_weaviate_filter(self, response):
        operator_map = {
            "LessThan": "less_than",
            "GreaterThan": "greater_than",
            "Equal": "equal"
        }

        filters = Filter.by_property("stock_status").equal("In stock")

        for item in response:
            path = item.get("path")
            operator = item.get("operator")
            value_number = item.get("valueNumber")
            value_string = item.get("valueString")

            if path in ["price", "rating"] and operator in operator_map:
                weaviate_operator = operator_map[operator]
                filter_condition = getattr(Filter.by_property(path), weaviate_operator)(float(value_number))
                filters &= filter_condition
            # elif path == "categories":
            #     filter_condition = Filter.by_property("categories").contains_any([value_string])
            #     filters &= filter_condition

        logging.info("Filter Generated Successfully")
        return filters


    def _extract_intent_features_and_category(self, response):
        """Extracts the intent from a given response."""
        intent = "unknown"
        features = []
        categories = []
        try:
            for item in response:
                path = item.get("path")
                if path and path == "intent":
                    intent = item.get("valueString", "unknown")
                elif path == "features":
                    feature = item.get("valueString")
                    if feature:
                        features.append(feature)
                elif path == "categories":
                    category = item.get("valueString")
                    if category:
                        categories.append(category)

            logging.info(f"Extracted intent: {intent}, features: {features}")
            return intent,features,categories

        except json.JSONDecodeError:
            logging.error("Invalid JSON format.")
            return "unknown",[]
    

    def extract_info_from_query(self, query, history):
        """Extract filters from the given query."""
        filters = Filter.by_property("stock_status").equal("In stock")
        intent = "unknown"
        try:       
            filters_str = self.llm_chain.run(query=query, history=history)

            cleaned_string = filters_str.strip("```").replace("json", "").strip()

            print(f"Query: {query}")
            print(f"Response: {cleaned_string}")

            valid_json = json.loads(cleaned_string)
            filters = self._convert_to_weaviate_filter(valid_json)
            intent, features, categories = self._extract_intent_features_and_category(valid_json)

            logging.info("Filters and Intent extracted")
            return filters, intent, features, categories

        except Exception as e:
            logging.error(f"Error extracting filters: {e}")
            return filters, intent, []

        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON response: {e}")
            return filters, intent, []