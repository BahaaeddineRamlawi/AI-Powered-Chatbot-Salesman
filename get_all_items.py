import weaviate

from utils import logging, config

class WeaviateClient:
    def __init__(self, collection_name):
        
        self.client = weaviate.connect_to_local()
        self.collection_name = collection_name
    

    def fetch_all_products(self):
        try:
            collection = self.client.collections.get(self.collection_name)

            response = collection.query.fetch_objects(
                limit=1000
            )

            if response.objects:
                logging.info(f"Successfully fetched {len(response.objects)} items from Weaviate.")
                return response.objects
            else:
                logging.warning(f"No items found in the {self.collection_name} class.")
                return []
        except Exception as e:
            logging.error(f"Error fetching items from Weaviate: {e}")
            return []
        finally:
            self.client.close()
    

    def log_results(self, all_products):
        if all_products:
            logging.info(f"Total items in the {self.collection_name} class: {len(all_products)}")
            for item in all_products:
                print(f"Product ID: {item.properties['product_id']}, Title: {item.properties['title']}, Price: {item.properties['price']}, Rating: {item.properties['rating']}")
        else:
            logging.info(f"No items found in the {self.collection_name} class.")


if __name__ == "__main__":
    client = WeaviateClient(config['weaviate']['collection_name'])
    all_products = client.fetch_all_products()
    client.log_results(all_products)