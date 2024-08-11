from abc import ABC, abstractmethod
from typing import Any, Dict, List


class DatabaseInterface(ABC):
    @abstractmethod
    def get_session(self):
        """Get a session to interact with the database."""
        raise NotImplementedError

    @abstractmethod
    def init_db(self) -> None:
        """Initialize the database with the necessary classes and properties.
        If the class already exists, it will not be created again."""
        raise NotImplementedError

    @abstractmethod
    def add_document(self, document: Dict[str, Any]) -> None:
        """Add a document to the database."""
        raise NotImplementedError

    @abstractmethod
    def load_documents(self, root_path: str) -> List[Dict[str, Any]]:
        """Load documents to the database from a given root path."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, limit: int) -> List[Any]:
        """Search for documents in the database."""
        raise NotImplementedError

    @abstractmethod
    def get_document_by_id(self, document_id: str) -> Dict[str, Any]:
        """Get a document by its ID."""
        raise NotImplementedError

    @abstractmethod
    def delete_document_by_id(self, document_id: str) -> None:
        """Delete a document by its ID."""
        raise NotImplementedError
