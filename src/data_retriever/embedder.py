from sentence_transformers import SentenceTransformer

from src.utils import logging, config

class ProductEmbedder:
    """Handles embedding generation for product data."""
    
    def __init__(self):
        """Initialize the embedding model based on the config file."""
        self.model_type = config['embedding']['model_type'] 
        self.model_name = config['embedding']['model_name']
        
        if self.model_type == "sentencetransformer":
            try:
                self.model = SentenceTransformer(self.model_name)
                logging.info(f"Embedding model '{self.model_name}' initialized successfully.")
            except Exception as e:
                logging.error(f"Error initializing embedding model: {e}")
                raise
        else:
            logging.error(f"Model type '{self.model_type}' is not supported yet.")
            raise ValueError(f"Unsupported model type: {self.model_type}")


    def generate_embeddings(self, df):
        """Generate and combine embeddings for the title and description."""
        try:
            logging.info("Generating embeddings...")

            df["combined_text"] = df.apply(lambda x: f"title: {x['title']} description: {x['description']} categories: {x['categories']}", axis=1)
            df["info_vector"] = df["combined_text"].apply(lambda x: self.model.encode(x))

            logging.info("Embeddings generated and validated successfully.")
            return df
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            raise