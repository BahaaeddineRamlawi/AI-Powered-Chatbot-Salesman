import csv
import os
import re
import time
from openai import AzureOpenAI

from src.utils import logging, config


class ProductCleaner:
    def __init__(self):
        self.input_file_path = config['data_file']['products_data_path']
        self.output_file_path = config['data_file']['cleaned_products_data_path']
        
        self.azure_cfg = config["azure_openai"]
        self.client = AzureOpenAI(
            api_version=self.azure_cfg["api_version"],
            azure_endpoint=self.azure_cfg["azure_endpoint"],
            api_key=self.azure_cfg["api_key"]
        )
        self.deployment = self.azure_cfg["azure_deployment"]


    def extract_prices(self, price_text):
        try:
            clean_text = price_text.replace('$', '').replace('\xa0', ' ')
            found = re.findall(r'\d+(?:\.\d+)?', clean_text)
            return [float(p) for p in found]
        except Exception as e:
            logging.error(f"Error extracting prices from '{price_text}': {e}")
            return []


    def parse_weight(self, weight):
        try:
            w = weight.lower().strip()
            match = re.search(r'(\d+(?:\.\d+)?)\s*(kg|g)', w)
            if not match:
                return float('inf')
            value, unit = float(match.group(1)), match.group(2)
            return value * 1000 if unit == 'kg' else value
        except Exception as e:
            logging.warning(f"Could not parse weight '{weight}': {e}")
            return float('inf')


    def weight_estimation(self, description):
        """Estimate weight from the description using Azure OpenAI if the weight is missing or invalid."""
        if not description or not description.strip():
            return None

        prompt = f"Estimate the weight of the following product: {description}. What is the weight in grams or kilograms?"

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a weight estimator and you will only return the weight without any details."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=256,
                    temperature=0,
                    top_p=1.0,
                )

                estimated_text = response.choices[0].message.content.strip()
                weight_match = re.search(r'(\d+(\.\d+)?\s?(kg|g|lbs|lb|oz))', estimated_text, re.IGNORECASE)

                if weight_match:
                    return weight_match.group(0)

                return None

            except Exception as e:
                logging.error(f"Error estimating weight using Azure OpenAI (attempt {attempt+1}): {e}")
                if '429' in str(e):
                    wait_time = 2 ** attempt
                    logging.info(f"Rate limit hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    break

        return None


    def expand_and_save(self):
        expanded_rows = []
        try:
            with open(self.input_file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    try:
                        all_prices = self.extract_prices(row.get('price', ''))
                        raw_weights = [w.strip() for w in row.get('weight', '').split(',') if w.strip()]
                        prices = sorted(all_prices)
                        weights = sorted(raw_weights, key=self.parse_weight)

                        if len(prices) > 1 and len(weights) == 1:
                            new_row = row.copy()
                            new_row['price'] = prices[0]
                            new_row['weight'] = weights[0]
                            new_row['description'] = row.get('description', '').strip() + f"\n\nOriginal prices: {', '.join(map(str, prices))}"
                            expanded_rows.append(new_row)

                        elif len(prices) == len(weights) and len(prices) > 1:
                            for price, weight in zip(prices, weights):
                                new_row = row.copy()
                                new_row['price'] = price
                                new_row['weight'] = weight
                                expanded_rows.append(new_row)

                        else:
                            row['price'] = float(prices[0]) if prices else ''
                            weight_field = row.get('weight', '').strip().lower()

                            if weight_field in ('', 'n/a', 'na', 'none'):
                                estimated = self.weight_estimation(row.get('description', ''))
                                if estimated:
                                    logging.info(f"Estimated weight for '{row['title']}': {estimated}")
                                    row['weight'] = estimated
                                else:
                                    logging.warning(f"Could not estimate weight for: {row['title']}")

                            expanded_rows.append(row)

                    except Exception as row_error:
                        logging.error(f"Failed to process row: {row}. Error: {row_error}")

        except FileNotFoundError:
            logging.error(f"Input file not found: {self.input_file_path}")
            return
        except Exception as e:
            logging.error(f"Unexpected error reading input CSV: {e}")
            return

        try:
            os.makedirs(os.path.dirname(self.output_file_path), exist_ok=True)
            with open(self.output_file_path, 'w', encoding='utf-8', newline='') as out_f:
                fieldnames = expanded_rows[0].keys() if expanded_rows else []
                writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(expanded_rows)

            logging.info(f"Expanded CSV saved to: {self.output_file_path}")

        except Exception as write_error:
            logging.error(f"Failed to write to output file {self.output_file_path}: {write_error}")