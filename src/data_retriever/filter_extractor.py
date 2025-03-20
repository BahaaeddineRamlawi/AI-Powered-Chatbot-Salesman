import json
from langchain_openai import AzureChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from weaviate.classes.query import Filter

from src.utils import logging, config


class QueryFilterExtractor:
    """Class to extract filters from product search queries using LLM."""
    
    def __init__(self):
        """Initialize the filter extractor with necessary configurations."""
        # self.llm_provider = config["llm"]["provider"]
        self.llm_provider = "mistral"
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

        self.prompt_template = """
        You are a filtering assistant for a search query. Your task is to extract filters from the following query.
        The filters may include price ranges, categories, ratings, and other relevant properties for product searches. Additionally, extract features such as "cheap", "gluten-free", etc.

        For each filter, provide the following keys:
        - "path": The path of the field (e.g., "price", "categories", "rating", "features")
        - "operator": The type of comparison (e.g., "LessThan", "GreaterThan", "Equal")
        - "valueNumber" or "valueString": The value to compare against (e.g., 50 for price, "electronics" for category, "gluten-free" for features)

        The following is the user query:
        {query}

        Based on this query, please provide the filters in the following JSON format:

        [
            {{  "path": "categories", "operator": "Equal", "valueString": "electronics" }},
            {{  "path": "rating", "operator": "GreaterThan", "valueNumber": 4 }},
            {{  "path": "features", "valueString": "gluten-free" }}
        ]

        If no filter is applicable, return an empty list. Include filters only when mentioned in the query.
        """
        
        self.prompt = PromptTemplate(input_variables=["query"], template=self.prompt_template)
        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)
    
    
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

            if path in ["price", "rating"] and operator in operator_map:
                weaviate_operator = operator_map[operator]
                filter_condition = getattr(Filter.by_property(path), weaviate_operator)(float(value_number))

                filters &= filter_condition

        logging.info(f"Generated Filter: {filters.__dict__ if hasattr(filters, '__dict__') else str(filters)}")
        return filters
    

    def extract_filters_from_query(self, query):
        """Extract filters from the given query."""
        try:
            logging.info(f"Extracting filters for query: {query}")
            
            filters_str = self.llm_chain.run(query=query)
            logging.info(f"Response returned from LLM.")

            cleaned_string = filters_str.strip("```").replace("json", "").strip()
            # print(f"Query: {query}")
            # print(f"Response: {cleaned_string}")
            valid_filters = json.loads(cleaned_string)
            
            return self._convert_to_weaviate_filter(valid_filters)

        except Exception as e:
            logging.error(f"Error extracting filters: {e}")
            return Filter.by_property("stock_status").equal("In stock")

        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON response: {e}")
            return Filter.by_property("stock_status").equal("In stock")