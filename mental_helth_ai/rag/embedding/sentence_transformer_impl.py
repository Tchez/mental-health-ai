from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer

from mental_helth_ai.rag.embedding.embedding_interface import (
    EmbeddingInterface,
)
from mental_helth_ai.settings import settings


class SentenceTransformerEmbedding(EmbeddingInterface):
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def generate_embeddings(
        self, documents: List[Dict[str, Any]]
    ) -> List[List[float]]:
        texts = [doc['page_content'] for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
