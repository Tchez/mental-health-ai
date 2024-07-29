from abc import ABC, abstractmethod
from typing import Any, Dict, List


class EmbeddingInterface(ABC):
    @abstractmethod
    def generate_embeddings(
        self, documents: List[Dict[str, Any]]
    ) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        raise NotImplementedError
