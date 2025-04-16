import pandas as pd
import sqlite3
import json
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import logging, config

class RecommendationHandler:
    def __init__(self):
        self.db_path = config['database']['history_name']
        self.products_data = pd.read_csv(config["input_file"]["cleaned_products_data_path"])
        self._load_ratings_from_db()
        self._prepare_data()
        self._compute_user_similarity()
        self._compute_item_similarity()


    def _load_ratings_from_db(self):
        try:
            logging.info("Loading user ratings from the database...")
            conn = sqlite3.connect(self.db_path)
            query = "SELECT user_id, product_list FROM user_history"
            df = pd.read_sql(query, conn)
            conn.close()

            ratings = []
            for _, row in df.iterrows():
                user_id = row["user_id"]
                product_list = json.loads(row["product_list"])
                ratings.extend([(user_id, item[0], item[1]) for item in product_list])

            self.ratings_data = pd.DataFrame(ratings, columns=["user_id", "product_id", "rating"])
            logging.info(f"Loaded {len(self.ratings_data)} ratings.")
        except Exception as e:
            logging.error(f"Error loading ratings from database: {e}")
            self.ratings_data = pd.DataFrame(columns=["user_id", "product_id", "rating"])


    def _prepare_data(self):
        logging.info("Preparing data and handling duplicates...")

        self.ratings_data = self.ratings_data.merge(
            self.products_data[["id"]], left_on="product_id", right_on="id", how="inner"
        )

        self.ratings_data = self.ratings_data.sort_values(by=["user_id", "product_id"]).drop_duplicates(
            subset=["user_id", "product_id"], keep="last"
        )

        self.user_item_matrix = self.ratings_data.pivot(index='user_id', columns='id', values='rating').fillna(0)
        logging.info(f"User-item matrix shape: {self.user_item_matrix.shape}")


    def _compute_user_similarity(self):
        logging.info("Computing user-user similarity...")
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )


    def _compute_item_similarity(self):
        logging.info("Computing item-item collaborative similarity...")
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity_df = pd.DataFrame(
            self.item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )


    def get_user_based_recommendations(self, user_id, top_k=10):
        rated_products = self.ratings_data[self.ratings_data["user_id"] == user_id]["id"].values

        if user_id not in self.user_similarity_df.index:
            logging.warning(f"User ID {user_id} not found in similarity index.")
            return []

        similar_users = self.user_similarity_df[user_id].sort_values(ascending=False).index[1:6]
        user_based_scores = self.user_item_matrix.loc[similar_users].mean().to_dict()
        user_based_scores = {
            pid: score for pid, score in user_based_scores.items()
            if pid not in rated_products
        }

        top_n_products = sorted(
            user_based_scores.items(), 
            key=lambda x: x[1], reverse=True
        )[:top_k]
        return self._format_recommendations(top_n_products)
    

    def get_item_based_recommendations(self, product_id, top_k=5):
        if product_id not in self.item_similarity_df.index:
            logging.error(f"Product ID {product_id} not found in item-item similarity index.")
            return []

        similarity_scores = self.item_similarity_df[product_id].drop(index=product_id)
        top_similar = similarity_scores.sort_values(ascending=False).head(top_k)

        product_score_pairs = list(top_similar.items())
        return self._format_recommendations(product_score_pairs)


    def _format_recommendations(self, recommendations):
        result = ""
        for i, (product_id, score) in enumerate(recommendations, 1):
            product_row = self.products_data[self.products_data["id"] == product_id]
            if product_row.empty:
                logging.warning(f"Product ID {product_id} not found in dataset, skipping.")
                continue

            logging.debug(f"Recommended product: {product_id}, Score: {score}")

            product_info = product_row.iloc[0]
            description = product_info["description"].rstrip()
            result += (
                f"Product {i}\n"
                f"Title: {product_info['title']}\n"
                f"Link: {product_info['link']}\n"
                f"Description: {description}\n"
                f"Categories: {product_info['categories']}\n"
                f"Price: ${product_info['price']}\n"
                f"Weight: {product_info['weight']}\n"
                f"Rating: {round(score, 2)}\n"
                f"Image URL: {product_info['image']}\n\n"
            )

        if not result:
            return "No recommended products found.\n"
        return result


