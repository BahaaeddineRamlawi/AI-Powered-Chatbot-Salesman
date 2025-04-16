import csv
import os
import re

from src.utils import logging, config

shop_url = config["website"]["name"]

def expand_products_by_weight_and_price(input_path, output_path):
    expanded_rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 1. Split raw strings into lists
            raw_prices = [p.strip() for p in re.split(r'–|-', row['price']) if p.strip()]
            raw_weights = [w.strip() for w in row['weight'].split(',') if w.strip()]

            # 2. Parse numeric values for sorting
            def parse_price(p):
                m = re.search(r'(\d+(?:\.\d+)?)', p.replace(',', ''))
                return float(m.group(1)) if m else float('inf')

            def parse_weight(w):
                w = w.lower().strip()
                match = re.search(r'(\d+(?:\.\d+)?)\s*(kg|g)', w)
                if not match:
                    return float('inf')  # Push unknowns to end
                value, unit = float(match.group(1)), match.group(2)
                return value * 1000 if unit == 'kg' else value

            # 3. Sort the variants by their numeric parts
            prices = sorted(raw_prices, key=parse_price)
            weights = sorted(raw_weights, key=parse_weight)

            # 4. If they match in length and >1, zip; otherwise, emit original
            if len(prices) == len(weights) and len(prices) > 1:
                for price, weight in zip(prices, weights):
                    new_row = row.copy()
                    new_row['price']  = price
                    new_row['weight'] = weight
                    expanded_rows.append(new_row)
            else:
                expanded_rows.append(row)

    # 5. Write out the expanded CSV
    fieldnames = reader.fieldnames
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(expanded_rows)

    print(f"✅ Expanded CSV saved to: {output_path}")

input_file = re.sub(r'[^a-zA-Z0-9]', '_', shop_url.strip('/')) + '.csv'
output_file = re.sub(r'[^a-zA-Z0-9]', '_', shop_url.strip('/')) + '_expanded.csv'
input_file_path = os.path.join('data', input_file)
output_file_path = os.path.join('data', output_file)

expand_products_by_weight_and_price(
    input_path=input_file_path,
    output_path=output_file_path
)
