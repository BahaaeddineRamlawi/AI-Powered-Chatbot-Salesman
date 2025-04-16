import requests
from bs4 import BeautifulSoup
import csv
import os
import re
import time

from src.utils import logging, config

start_time = time.time()
shop_url = config["website"]["name"]
all_products = []

page = 1

def extract_weight_from_soup(soup, fallback_text):
    try:
        def filter_kg_g(text):
            return ', '.join(re.findall(r'\b\d+(?:\.\d+)?\s?(?:kg|g)\b', text, flags=re.IGNORECASE))

        # Priority 1: Check known weight-related <select> elements
        for select_id in ['sold-and-packed-in', 'packed-in', 'size']:
            select = soup.find('select', id=select_id)
            if select:
                options = select.find_all('option')
                weights = [
                    filter_kg_g(opt.text.strip())
                    for opt in options
                    if opt.get('value') and 'choose an option' not in opt.text.lower()
                ]
                weights = [w for w in weights if w]  # Remove empty strings
                if weights:
                    return ', '.join(weights)

        # Priority 2: Check fallback text for SOLD IN
        sold_in_match = re.search(r"SOLD IN:\s*(.*?)(?:\n|$)", fallback_text, re.IGNORECASE)
        if sold_in_match:
            sold_in_text = sold_in_match.group(1)
            filtered = filter_kg_g(sold_in_text)
            if filtered:
                return filtered
    except Exception as e:
        logging.error(f"Weight extraction error: {e}")
    return 'N/A'

while True:
    url = f"{shop_url}/page/{page}/" if page > 1 else shop_url
    logging.info(f"Fetching: {url}")

    try:
        response = requests.get(url)
        response.encoding = 'utf-8'
        response.raise_for_status()
    except requests.RequestException as e:
        logging.warning(f"Page fetch failed at {url}: {e}")
        break

    soup = BeautifulSoup(response.text, 'html.parser')
    products = soup.find_all(['li', 'div'], class_='product')

    if not products:
        logging.info(f"No products found on page {page}. Ending pagination.")
        break

    for product in products:
        try:
            pid = product.get('data-id', 'N/A')

            title_elem = product.find('h3', class_='wd-entities-title')
            product_title = title_elem.find('a').text.strip() if title_elem and title_elem.find('a') else 'N/A'
            product_link = title_elem.find('a')['href'] if title_elem and title_elem.find('a') else 'N/A'

            rating_div = product.find('div', class_='jdgm-prev-badge')
            rating_raw = rating_div.get('data-average-rating', '') if rating_div else ''
            try:
                rating_text = float(rating_raw) if rating_raw else 'N/A'
            except ValueError:
                rating_text = 'N/A'

            description_elem = product.find('div', class_='product-meta')
            description_text = description_elem.text.strip() if description_elem else 'N/A'

            price_elem = product.find('span', class_='price')
            price_text = price_elem.text.strip() if price_elem else 'N/A'

            image_elem = product.find('img')
            image_src = image_elem['src'] if image_elem else 'N/A'

            out_of_stock = product.find('span', class_='out-of-stock')
            stock_status = 'SOLD OUT' if out_of_stock else 'IN STOCK'

            full_description = 'N/A'
            categories_list = []
            weight_value = 'N/A'

            if product_link != 'N/A':
                try:
                    product_response = requests.get(product_link)
                    product_response.encoding = 'utf-8'

                    if product_response.status_code == 200:
                        product_soup = BeautifulSoup(product_response.text, 'html.parser')

                        desc_container = product_soup.find('div', id='tab-description')
                        desc_text = desc_container.get_text(separator="\n", strip=True) if desc_container else ''

                        add_info_container = product_soup.find('div', id='tab-additional_information')
                        add_info_text = add_info_container.get_text(separator="\n", strip=True) if add_info_container else ''

                        full_description = (desc_text + "\n\n" + add_info_text).strip() or 'N/A'
                        weight_value = extract_weight_from_soup(product_soup, full_description)

                        categories_section = product_soup.select_one('.product_meta .posted_in')
                        if categories_section:
                            categories_links = categories_section.find_all('a')
                            categories_list = [link.text.strip() for link in categories_links]
                    else:
                        logging.warning(f"Failed to fetch product page: {product_link} - Status: {product_response.status_code}")
                except Exception as e:
                    logging.error(f"Error fetching product details from {product_link}: {e}")

                time.sleep(0.5)

            all_products.append([
                pid,
                product_title,
                price_text,
                product_link,
                ', '.join(categories_list) if categories_list else 'N/A',
                full_description,
                rating_text,
                weight_value,
                image_src,
                stock_status
            ])
        except Exception as e:
            logging.error(f"❌ Error parsing product block: {e}")

    logging.info(f"Page {page} scraped: {len(products)} products")
    page += 1

# Save to CSV
os.makedirs('data', exist_ok=True)
filename = re.sub(r'[^a-zA-Z0-9]', '_', shop_url.strip('/')) + '.csv'
file_path = os.path.join('data', filename)

try:
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'product_id', 'title', 'price', 'link', 'categories',
            'description', 'rating', 'weight', 'image', 'stock_status'
        ])
        writer.writerows(all_products)
    logging.info(f"Scraping complete. Saved {len(all_products)} products to {file_path}")
except Exception as e:
    logging.error(f"Failed to write CSV: {e}")

end_time = time.time()
minutes, seconds = divmod(end_time - start_time, 60)
logging.info(f"Total time taken: {int(minutes)} minutes and {int(seconds)} seconds")
