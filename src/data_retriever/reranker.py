from flashrank import Ranker, RerankRequest
from src.utils import logging
from sentence_transformers import util
import re

INTENT_EXAMPLES = {
    "asc": [
        "show me the cheapest products",
        "I want affordable items",
        "give me the least expensive products",
        "I want low price options",
        "show lowest prices",
        "cheap products",
        "affordable gifts"
    ],
    "desc": [
        "show me the most expensive products",
        "I want premium or luxury items",
        "give me your most expensive product",
        "show highest price items",
        "what's your priciest product",
        "expensive products",
        "expensive gifts",
        "premium gifts",
        "show me expensive gifts"
    ]
}

def normalize_text(text):
    """Lowercase and strip punctuation from text."""
    return re.sub(r'[^\w\s]', '', text.lower().strip())

class RerankedResponse:
    def __init__(self, embedding_model):
        """
        Initialize RerankedResponse with FlashRank Ranker and embedding model.
        Cache intent embeddings for reuse.
        """
        self.ranker = Ranker()
        self.objects = None
        self.embedding_model = embedding_model
        self.intent_example_embeddings = {
            intent: self.embedding_model.encode(examples, convert_to_tensor=True)
            for intent, examples in INTENT_EXAMPLES.items()
        }

    def process_objects(self, objects, limit=None):
        """
        Process the documents (apply limit and create RerankedObject).
        """
        if limit is not None:
            objects = objects[:limit]

        self.objects = [
            type('RerankedObject', (), {'properties': doc}) for doc in objects
        ]

    def get_price_sort_intent(self, query_text):
        """
        Infer user intent from query using semantic similarity (max similarity).
        Returns 'asc', 'desc', or None.
        """
        try:
            query_text = normalize_text(query_text)
            query_embedding = self.embedding_model.encode(query_text, convert_to_tensor=True)

            best_score = -1
            best_intent = None

            for intent, example_embeddings in self.intent_example_embeddings.items():
                similarity = util.cos_sim(query_embedding, example_embeddings)
                max_score = similarity.max().item()

                if max_score > best_score:
                    best_score = max_score
                    best_intent = intent

            return best_intent if best_score > 0.45 else None

        except Exception as e:
            logging.error(f"Intent detection failed: {e}")
            return None

    def rerank_results(self, query, documents, return_top_k=8):
        """
        Rerank search results using FlashRank, then apply optional price-based sorting.
        """
        try:
            rerank_input = RerankRequest(
                query=query,
                passages=[
                    {
                        "text": " ".join(filter(None, [
                            doc.get('title', ''),
                            doc.get('description', ''),
                            " ".join(doc.get('categories', [])) if isinstance(doc.get('categories'), list) else doc.get('categories', '')
                        ])).strip(),
                        "id": f"{doc.get('product_id', '')}_{doc.get('weight', '').strip()}"
                    } for doc in documents
                ]
            )

            reranked_results = self.ranker.rerank(rerank_input)

            scored_docs = []
            for ranked_result in reranked_results:
                reranked_id = ranked_result.get('id')
                if "_" in reranked_id:
                    product_id, weight = reranked_id.split("_", 1)
                else:
                    product_id, weight = reranked_id, None

                matching_doc = next(
                    (doc for doc in documents
                    if doc.get('product_id') == product_id and doc.get('weight', '').strip() == weight),
                    None
                )

                if matching_doc:
                    matching_doc['rerank_score'] = ranked_result.get('score', 0)
                    scored_docs.append(matching_doc)

            if not scored_docs:
                logging.warning("No documents matched reranked results.")
                return documents[:return_top_k]

            sort_direction = self.get_price_sort_intent(query)
            logging.info(f"Inferred intent: {sort_direction}")

            if sort_direction in {"asc", "desc"}:
                scored_docs = sorted(
                    scored_docs,
                    key=lambda d: float(str(d.get('price', 0)).replace("$", "").strip()),
                    reverse=(sort_direction == "desc")
                )
                logging.info(f"Applied price sorting to reranked results: {sort_direction.upper()}")
            else:
                logging.info("No price sorting applied.")

            final_docs = scored_docs[:return_top_k]
            logging.info(f"Returning {len(final_docs)} items")

            return final_docs

        except Exception as e:
            logging.error(f"Reranking failed: {e}")
            return documents[:return_top_k]