from flashrank import Ranker, RerankRequest
from typing import Any, Dict, List
from src.utils import logging, config

class RerankedResponse:
    def __init__(self):
        """
        Initialize RerankedResponse with optional limit on number of objects
        and initialize the Ranker.

        :param objects: List of document objects
        :param limit: Maximum number of objects to include (optional)
        """
        self.ranker = Ranker()
        self.objects = None

    def process_objects(self, objects, limit=None):
        """
        Process the documents (apply limit and create RerankedObject).

        :param objects: List of document objects
        :param limit: Optional limit on the number of objects
        """
        if limit is not None:
            objects = objects[:limit]

        # Convert objects into RerankedObject
        self.objects = [
            type('RerankedObject', (), {'properties': doc}) for doc in objects
        ]

    def rerank_results(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank search results using FlashRank

        :param query: Search query string
        :param documents: List of document dictionaries to rerank
        :return: Reranked list of documents
        """
        try:
            rerank_input = RerankRequest(
                query=query,
                passages=[
                    {
                        "text": " ".join(filter(None, [
                            doc.get('title', ''),
                            doc.get('description', ''),
                            doc.get('categories', '')
                        ])).strip(),
                        "id": doc.get('product_id', '')
                    } for doc in documents
                ]
            )
            
            # Perform reranking
            reranked_results = self.ranker.rerank(rerank_input)
            
            # Map reranked results back to original documents
            reranked_docs = []
            for ranked_result in reranked_results:
                # Find the original document that matches this ranked result
                matching_doc = next(
                    (doc for doc in documents if 
                     ranked_result.get('id') == doc.get('product_id')), 
                    None
                )
                
                if matching_doc:
                    matching_doc['rerank_score'] = ranked_result.get('score', 0)
                    reranked_docs.append(matching_doc)
            
            # If no matches found, fall back to original documents
            if not reranked_docs:
                reranked_docs = documents
            return reranked_docs
        
        except Exception as e:
            logging.error(f"Reranking failed: {e}")
            return documents
