from weaviate.collections.classes.filters import Filter

from .weaviate import WeaviateHandler
from .recommendation import RecommendationHandler

from src.utils import logging

class MarketingStrategies:
    def __init__(self, weaviate_handler, recommendation_handler):
        self.weaviate_handler = weaviate_handler
        self.recommendation_handler = recommendation_handler

    def get_cross_sell_recommendations(self, base_product, limit=5):
        """
        Recommend similar products using item-based collaborative filtering.
        """
        try:
            product_id = base_product.get("product_id")
            if not product_id:
                logging.warning("Base product does not have a product_id.")
                return []

            logging.info(f"Fetching cross-sell recommendations for product_id={product_id}")
            return self.recommendation_handler.get_item_based_recommendations(
                product_id=int(product_id), top_k=limit
            )
        except Exception as e:
            logging.error(f"Error in get_cross_sell_recommendations: {e}", exc_info=True)
            return []

    def get_up_sell_recommendations(self, base_product, limit=5):
        """
        Recommend more expensive products in the same category (up-sell).
        """
        try:
            base_price = base_product.get("price", 0)
            categories = base_product.get("categories", "")
            filters = (
                Filter.by_property("price").greater_than(base_price) &
                Filter.by_property("categories").contains_all(categories)
            )

            title = base_product.get("title", "")
            description = base_product.get("description", "")
            query_string = f"{title}. {description}"

            logging.info(f"Fetching up-sell recommendations for '{title}' with price > {base_price}")
            results, _ = self.weaviate_handler.hybrid_search(
                query=query_string,
                alpha=0.3,
                filters=filters,
                limit=limit
            )
            return results
        except Exception as e:
            logging.error(f"Error in get_up_sell_recommendations: {e}", exc_info=True)
            return []

if __name__ == "__main__":
    try:
        weaviate_handler = WeaviateHandler()
        recommendation_handler = RecommendationHandler()
        strategies = MarketingStrategies(weaviate_handler, recommendation_handler)

        query = "4-NRGY Protein Nuts"
        results, base_product = strategies.weaviate_handler.hybrid_search(query=query, alpha=0.3, limit=1)

        if not base_product:
            logging.info("No base product found.")
            print("No base product found.")
        else:
            print("Main Product:")
            print(base_product)

            cross_sell = strategies.get_cross_sell_recommendations(base_product)
            up_sell = strategies.get_up_sell_recommendations(base_product)

            print("\nCross-Sell Recommendations:\n")
            print(cross_sell)

            print("\nUp-Sell Recommendations:\n")
            print(up_sell)
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
    finally:
        try:
            weaviate_handler.close()
        except Exception as e:
            logging.error(f"Error closing Weaviate handler: {e}", exc_info=True)