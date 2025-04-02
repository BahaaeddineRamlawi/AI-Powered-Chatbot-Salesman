import pandas as pd
import sqlite3
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import logging, config

class RecommendationHandler:
    def __init__(self):
        self.db_path = config['database']['history_name']
        self.products_data = pd.read_csv(config["input_file"]["cleaned_products_data_path"])
        self._load_ratings_from_db()
        self._prepare_data()
        self._compute_user_similarity()
        self._compute_content_similarity()
    
    def _load_ratings_from_db(self):
        """Load user ratings from the SQLite database."""
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
            self.ratings_data = pd.DataFrame(columns=["user_id", "product_id", "rating"])  # Ensure empty DataFrame on failure
    
    def _prepare_data(self):
        """Prepare data by merging and handling duplicate ratings."""
        logging.info("Preparing data and handling duplicates...")
        
        self.ratings_data = self.ratings_data.merge(
            self.products_data[["id"]], left_on="product_id", right_on="id", how="inner"
        )

        # Handle duplicate (user_id, product_id) entries by keeping the latest rating
        self.ratings_data = self.ratings_data.sort_values(by=["user_id", "product_id"]).drop_duplicates(
            subset=["user_id", "product_id"], keep="last"
        )

        # Pivot table for user-item interactions
        self.user_item_matrix = self.ratings_data.pivot(index='user_id', columns='id', values='rating').fillna(0)

        logging.info(f"User-item matrix shape: {self.user_item_matrix.shape}")
    
    def _compute_user_similarity(self):
        """Compute user-user similarity using cosine similarity."""
        logging.info("Computing user-user similarity...")
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity, 
            index=self.user_item_matrix.index, 
            columns=self.user_item_matrix.index
        )
    
    def _compute_content_similarity(self):
        """Compute content-based similarity using TF-IDF on product descriptions."""
        logging.info("Computing content-based similarity...")
        self.products_data["description"] = self.products_data["description"].fillna("")
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(self.products_data["description"])
        self.content_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Create a mapping of product ID to index
        self.product_id_to_index = {
            product_id: idx 
            for idx, product_id in enumerate(self.products_data["id"].values)
        }

    def get_hybrid_recommendations(self, user_id, product_id):
        """
        Generate hybrid recommendations for a given user based on a specific product.
        
        Args:
            user_id (int): The ID of the user for whom recommendations are generated.
            product_id (int): The ID of the product to find similar items to. If None, only collaborative filtering is used.

        Returns:
            List[Dict]: A list of dictionaries containing recommended product details.
        """
        alpha = config['recommendation_system']['alpha']
        n = config['recommendation_system']['n']
        
        if product_id and product_id not in self.product_id_to_index:
            logging.error(f"Product ID {product_id} not found in product dataset.")
            return []

        all_product_ids = self.ratings_data["id"].unique()
        rated_products = self.ratings_data[self.ratings_data["user_id"] == user_id]["id"].values

        # Collaborative Filtering - User-Based
        user_based_scores = {}
        if user_id in self.user_similarity_df.index:
            similar_users = self.user_similarity_df[user_id].sort_values(ascending=False).index[1:6]  # Top 5 similar users
            user_based_scores = self.user_item_matrix.loc[similar_users].mean().to_dict()

        # Content-Based Filtering - Find similar items to the given product (only if product_id is provided)
        cb_predictions = {}
        if product_id:  # Only apply content-based filtering if product_id is provided
            product_idx = self.product_id_to_index[product_id]
            
            for candidate_product_id in all_product_ids:
                if candidate_product_id != product_id and candidate_product_id not in rated_products:
                    if candidate_product_id in self.product_id_to_index:
                        similarity_score = self.content_sim_matrix[product_idx, self.product_id_to_index[candidate_product_id]]
                        cb_predictions[candidate_product_id] = similarity_score * 5  # Scale similarity score

        # Hybrid Recommendation - Combining CF and CB
        hybrid_scores = {
            candidate_product_id: alpha * user_based_scores.get(candidate_product_id, 0) + 
                                (1 - alpha) * cb_predictions.get(candidate_product_id, 0)
            for candidate_product_id in all_product_ids 
            if candidate_product_id != product_id and candidate_product_id not in rated_products
        }

        # Get top N recommendations
        top_n_products = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        recommended_products = []
        for candidate_product_id, score in top_n_products:
            product_row = self.products_data[self.products_data["id"] == candidate_product_id]

            if product_row.empty:
                logging.warning(f"Product ID {candidate_product_id} not found in dataset, skipping.")
                continue

            product_info = product_row.iloc[0]  # Extract first matching row safely
            
            recommended_products.append({
                "id": candidate_product_id,
                "title": product_info["title"],
                "description": product_info["description"][:100] + "..." if len(product_info["description"]) > 100 else product_info["description"],
                "price": product_info["price"],
                "weight": product_info["weight"],
                "image": product_info["image"],
                "link": product_info["link"],
                "score": round(score, 2)
            })

        logging.info(f"Generated {len(recommended_products)} recommendations for user {user_id} based on product {product_id if product_id else 'N/A'}")
        return recommended_products

