from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class DatabaseInterface(ABC):
    @abstractmethod
    def load_index(self):
        """Load the index from a file."""
        raise NotImplementedError

    @abstractmethod
    def save_index(self):
        """Save the index to a file."""
        raise NotImplementedError

    @abstractmethod
    def create_index(self, embeddings: List[List[float]]):
        """Create a new index with the given embeddings."""
        raise NotImplementedError

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the index."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search the index for the given query."""
        raise NotImplementedError
