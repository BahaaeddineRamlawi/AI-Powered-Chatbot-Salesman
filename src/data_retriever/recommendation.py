import pandas as pd
import sqlite3
import json
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from src.utils import logging, config

class RecommendationHandler:
    def __init__(self):
        self.db_path = config['database']['history_name']
        self.products_data = pd.read_csv(config["input_file"]["cleaned_products_data_path"])
        self._load_ratings_from_db()
        self._prepare_data()
        self._train_model()
        self._compute_content_similarity()
    
    def _load_ratings_from_db(self):
        """Load user ratings from the SQLite database."""
        try:
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
        except Exception as e:
            logging.error(f"Error loading ratings from database: {e}")
    
    def _prepare_data(self):
        self.ratings_data = self.ratings_data.merge(
            self.products_data[["id"]], left_on="product_id", right_on="id", how="inner"
        )
        
    def _train_model(self):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.ratings_data[["user_id", "id", "rating"]], reader)
        
        param_grid = {"n_factors": [50, 100], "reg_all": [0.02, 0.1], "lr_all": [0.005, 0.01]}
        gs = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=3)
        gs.fit(data)
        
        self.algo = gs.best_estimator["rmse"]
        trainset = data.build_full_trainset()
        self.algo.fit(trainset)
    
    def _compute_content_similarity(self):
        self.products_data["description"] = self.products_data["description"].fillna("")
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(self.products_data["description"])
        self.content_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        # Create a mapping of product ID to index in the content similarity matrix
        self.product_id_to_index = {product_id: idx for idx, product_id in enumerate(self.products_data["id"].values)}
    
    def get_hybrid_recommendations(self, user_id, n=5, alpha=0.7):
        all_product_ids = self.ratings_data["id"].unique()
        rated_products = self.ratings_data[self.ratings_data["user_id"] == user_id]["id"]
        cf_predictions = {
            product_id: self.algo.predict(user_id, product_id).est
            for product_id in all_product_ids if product_id not in rated_products.values
        }
        
        cb_predictions = {}
        user_rated_product_ids = rated_products.values
        for product_id in all_product_ids:
            if product_id not in user_rated_product_ids:
                # Use the product_id_to_index mapping to safely access the content similarity matrix
                if product_id in self.product_id_to_index:
                    content_scores = [
                        self.content_sim_matrix[self.product_id_to_index[product_id], self.product_id_to_index[rated_id]]
                        for rated_id in user_rated_product_ids if rated_id in self.product_id_to_index
                    ]
                    cb_predictions[product_id] = (sum(content_scores) / len(content_scores) if content_scores else 0.0) * 5
        
        hybrid_scores = {
            product_id: alpha * cf_predictions.get(product_id, 0) + (1 - alpha) * cb_predictions.get(product_id, 0)
            for product_id in all_product_ids if product_id not in rated_products.values
        }
        
        top_n_products = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        recommended_products = [(self.products_data[self.products_data["id"] == product_id]["title"].values[0], score)
                                 for product_id, score in top_n_products]
        return recommended_products
