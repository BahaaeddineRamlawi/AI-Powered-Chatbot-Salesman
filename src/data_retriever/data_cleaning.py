import pandas as pd
import re
from mistralai import Mistral
import time
import numpy as np

from ..utils.logger_setup import logging
from ..utils.config_loader import config

class ProductDataCleaner:
    def __init__(self, file_path):
        self.file_path = file_path
        self.client = Mistral(api_key=config['mistral']['api_key'])
        self.model = config['mistral']['model']

    def weight_estimation(self, description):
        """Estimate weight from the description using Mistral API if the weight is missing or invalid."""
        if pd.isna(description):
            return None
        
        prompt = f"Estimate the weight of the following product: {description} What is the weight in grams or kilograms?"
        
        attempt = 0
        while attempt < 3:  # Retry 3 times before failing
            try:
                chat_response = self.client.chat.complete(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": prompt,
                    }]
                )
                
                estimated_weight = chat_response.choices[0].message.content
                weight_match = re.search(r'(\d+(\.\d+)?\s?(kg|g|lbs|lb|oz))', estimated_weight, re.IGNORECASE)
                
                if weight_match:
                    return weight_match.group(0)
                return None
            except Exception as e:
                logging.error(f"Error estimating weight using Mistral API (attempt {attempt+1}): {e}")
                if '429' in str(e):
                    attempt += 1
                    wait_time = (2 ** attempt)
                    logging.info(f"Rate limit exceeded. Retrying after {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    break
        return None

    def extract_weight(self, description):
        """Extract weight from the description using regex, if available."""
        if pd.isna(description):
            return None
        weight_pattern = r'(\d+(\.\d+)?\s?(kg|g|lbs|lb|oz))'
        matches = re.findall(weight_pattern, description, re.IGNORECASE)
        return ', '.join(match[0] for match in matches) if matches else None

    def extract_rating(self, rating_str):
        """Extract the first numeric value from the rating string or return NaN if empty."""
        match = re.search(r"(\d+(\.\d+)?)", str(rating_str))
        return float(match.group(1)) if match else np.nan

    def clean_data(self, df):
        """Clean the product data to ensure no NaN, inf, or invalid values."""
        df["price"] = pd.to_numeric(df["price"].astype(str).str.replace(r'[ $,]', '', regex=True), errors='coerce').apply(lambda x: np.nan if pd.isna(x) else x)
        
        df["rating"] = df["rating"].apply(self.extract_rating)

        def get_valid_weight(row):
            if pd.notna(row["weight"]) and re.match(r'\d+(\.\d+)?\s?(kg|g|lbs|lb|oz)', str(row["weight"]), re.IGNORECASE):
                return row["weight"]
            extracted_weight = self.extract_weight(row["description"])
            if extracted_weight:
                return extracted_weight
            return self.weight_estimation(row["description"])

        df["weight"] = df.apply(lambda row: get_valid_weight(row), axis=1)

        logging.info("Data cleaned successfully.")
        return df

    def process_file(self):
        """Load, clean, and save the cleaned CSV file with UTF-8 encoding."""
        try:
            df = pd.read_csv(self.file_path, encoding="ISO-8859-1")

            logging.info("File loaded successfully.")

            df = self.clean_data(df)

            cleaned_file_path = config['output_file']['cleaned_products_data_path']
            df.to_csv(cleaned_file_path, index=False, encoding="utf-8", na_rep="NaN")

            logging.info(f"Cleaned data saved to {cleaned_file_path}")
            return cleaned_file_path
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            return None

cleaner = ProductDataCleaner(config['input_file']['products_data_path'])
cleaned_file = cleaner.process_file()
