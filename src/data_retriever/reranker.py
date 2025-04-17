from flashrank import Ranker, RerankRequest
from src.utils import logging

class RerankedResponse:
    def __init__(self):
        """
        Initialize RerankedResponse with optional limit on number of objects
        and initialize the Ranker.
        """
        self.ranker = Ranker()
        self.objects = None

    def process_objects(self, objects, limit=None):
        """
        Process the documents (apply limit and create RerankedObject).
        """
        if limit is not None:
            objects = objects[:limit]

        self.objects = [
            type('RerankedObject', (), {'properties': doc}) for doc in objects
        ]

    def rerank_results(self, query, documents):
        """
        Rerank search results using FlashRank, handling variants properly.
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

            reranked_docs = []
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
                    reranked_docs.append(matching_doc)

            if not reranked_docs:
                reranked_docs = documents

            return reranked_docs

        except Exception as e:
            logging.error(f"Reranking failed: {e}")
            return documents
