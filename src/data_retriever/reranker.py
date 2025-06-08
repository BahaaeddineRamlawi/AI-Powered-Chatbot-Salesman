import re
from flashrank import Ranker, RerankRequest
from sentence_transformers import util
from src.utils import logging

SEMANTIC_RULES = {
    "asc": [
        "show me the cheapest products",
        "I want affordable items",
        "give me the least expensive products",
        "I want low price options",
        "show lowest prices",
        "cheap products",
        "affordable gifts",
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

KEYWORD_RULES = {
    "asc": ["cheapest", "low price", "affordable", "cheap", "lowest", "least expensive"],
    "desc": ["most expensive", "luxury", "premium", "highest price", "priciest", "more expensive"]
}


def normalize_text(text):
    """Lowercase and strip punctuation from text."""
    return re.sub(r'[^\w\s]', '', text.lower().strip())


class RerankedResponse:
    def __init__(self, embedding_model):
        self.ranker = Ranker()
        self.objects = None
        self.embedding_model = embedding_model

        self.intent_example_embeddings = {
            intent: self.embedding_model.encode(examples, convert_to_tensor=True)
            for intent, examples in SEMANTIC_RULES.items()
        }


    def process_objects(self, objects, limit=None):
        if limit is not None:
            objects = objects[:limit]
        self.objects = [type('RerankedObject', (), {'properties': doc}) for doc in objects]


    def get_price_sort_intent(self, query_text, alpha=0.45):
        """
        Detect user intent using hybrid scoring: keyword + semantic.
        """
        try:
            query_text = normalize_text(query_text)

            query_embedding = self.embedding_model.encode(query_text, convert_to_tensor=True)
            semantic_scores = {}

            for intent, example_embeddings in self.intent_example_embeddings.items():
                similarity = util.cos_sim(query_embedding, example_embeddings)
                max_score = similarity.max().item()
                semantic_scores[intent] = max_score

            logging.info(f"[INTENT] Semantic scores: {semantic_scores}")

            keyword_scores = {intent: 0 for intent in SEMANTIC_RULES}
            for intent, keywords in KEYWORD_RULES.items():
                for keyword in keywords:
                    if keyword in query_text:
                        keyword_scores[intent] += 1

            max_keyword_score = max(keyword_scores.values()) or 1
            keyword_scores = {
                intent: score / max_keyword_score
                for intent, score in keyword_scores.items()
            }

            logging.info(f"[INTENT] Keyword scores (normalized): {keyword_scores}")

            hybrid_scores = {
                intent: alpha * semantic_scores.get(intent, 0) + (1 - alpha) * keyword_scores.get(intent, 0)
                for intent in SEMANTIC_RULES
            }

            logging.info(f"[INTENT] Hybrid scores (alpha={alpha}): {hybrid_scores}")

            best_intent = max(hybrid_scores, key=hybrid_scores.get)
            best_score = hybrid_scores[best_intent]

            logging.info(f"[INTENT] Best intent: {best_intent} (score: {best_score:.2f})")

            return best_intent if best_score > 0.45 else None

        except Exception as e:
            logging.error(f"[INTENT] Hybrid intent detection failed: {e}")
            return None


    def rerank_results(self, query, documents, return_top_k=8):
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
                logging.warning("[RERANK] No documents matched reranked results.")
                return documents[:return_top_k]

            sort_direction = self.get_price_sort_intent(query)
            logging.info(f"[RERANK] Inferred price sorting intent: {sort_direction}")

            if sort_direction in {"asc", "desc"}:
                scored_docs = sorted(
                    scored_docs,
                    key=lambda d: float(str(d.get('price', 0)).replace("$", "").strip()),
                    reverse=(sort_direction == "desc")
                )
                logging.info(f"[RERANK] Applied price sorting: {sort_direction.upper()}")
            else:
                logging.info("[RERANK] No price sorting applied.")

            final_docs = scored_docs[:return_top_k]
            logging.info(f"[RERANK] Returning {len(final_docs)} items")

            return final_docs

        except Exception as e:
            logging.error(f"[RERANK] Failed to rerank results: {e}")
            return documents[:return_top_k]
