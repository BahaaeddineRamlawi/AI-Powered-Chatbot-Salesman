import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import logging, config

class RecommendationHandler:
    def __init__(self):
        try:
            self.products_df = pd.read_csv(config["input_file"]["cleaned_products_data_path"], encoding="utf-8")
            self.ratings_df = pd.read_csv(config["input_file"]["user_rating_datapath"], encoding="utf-8")
            logging.info("Datasets loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading datasets: {e}")
            raise

        self.ratings_df["product_id"] = self.ratings_df["product_id"].astype(str)
        self.products_df["id"] = self.products_df["id"].astype(str)

        self.ratings_df = self.ratings_df.groupby(["user_id", "product_id"], as_index=False)["rating"].mean()

        self.product_mapping = dict(zip(self.products_df["id"], self.products_df["title"]))

        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.products_df["description"].fillna(""))

        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        self.user_item_matrix = self.ratings_df.pivot(index="user_id", columns="product_id", values="rating").fillna(0)
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_sim_df = pd.DataFrame(self.user_similarity, index=self.user_item_matrix.index, columns=self.user_item_matrix.index)

    def get_content_based_recommendations(self, product_id, num_recommendations=5):
        """Get recommendations based on product similarity."""
        try:
            product_index = self.products_df.index[self.products_df["id"] == product_id].tolist()[0]
            similarity_scores = list(enumerate(self.cosine_sim[product_index]))
            sorted_similar_products = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
            return [self.products_df.iloc[i[0]]["id"] for i in sorted_similar_products]
        except IndexError:
            logging.error(f"Product ID {product_id} not found.")
            return []
        except Exception as e:
            logging.error(f"Error in content-based recommendation: {e}")
            return []

    def get_collaborative_recommendations(self, user_id, num_recommendations=5):
        """Get recommendations based on user similarity."""
        try:
            if user_id not in self.user_sim_df.index:
                logging.warning(f"User ID {user_id} not found.")
                return []

            similar_users = self.user_sim_df[user_id].sort_values(ascending=False)[1:6].index
            similar_users_ratings = self.user_item_matrix.loc[similar_users].mean()

            already_rated = self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index
            recommendations = similar_users_ratings.drop(already_rated).sort_values(ascending=False).index[:num_recommendations]

            return recommendations.tolist()
        except Exception as e:
            logging.error(f"Error in collaborative recommendation for User ID {user_id}: {e}")
            return []

    def hybrid_recommendation(self, user_id, product_id, num_recommendations=5):
        """Combines content-based and collaborative filtering."""
        try:
            content_recs = self.get_content_based_recommendations(product_id, num_recommendations)
            collab_recs = self.get_collaborative_recommendations(user_id, num_recommendations)

            hybrid_recs = list(set(content_recs + collab_recs))[:num_recommendations]

            if not hybrid_recs:
                return "No recommendations available."

            product_names = [self.product_mapping.get(pid, "Unknown Product") for pid in hybrid_recs]
            return f"If you bought **{self.product_mapping.get(product_id, 'this product')}**, you should try:\n- " + "\n- ".join(product_names)
        except Exception as e:
            logging.error(f"Error generating hybrid recommendations for User ID {user_id} and Product ID {product_id}: {e}")
            return "Error: Recommendation generation failed."