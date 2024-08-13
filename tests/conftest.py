import pytest

from mental_helth_ai.rag.database.weaviate_impl import WeaviateClient


@pytest.fixture
def weaviate_client() -> WeaviateClient:
    """Fixture to create a WeaviateClient instance."""
    return WeaviateClient()


@pytest.fixture
def sample_document():
    """Fixture to provide a sample valid document."""
    return {
        'title': 'Sample Document',
        'page_content': 'This is a sample document content.',
        'metadata': {
            'type': 'article',
            'source': 'https://example.com',
            'page_number': 1,
            'source_description': 'Example site',
            'date': '2023-01-01T00:00:00Z',
        },
    }


@pytest.fixture
def invalid_document():
    """Fixture to provide an invalid document (missing title)."""
    return {
        'page_content': 'This document has no title.',
        'metadata': {
            'type': 'article',
            'source': 'https://example.com',
            'page_number': 1,
            'source_description': 'Example site',
            'date': '2023-01-01T00:00:00Z',
        },
    }


# TODO: Utilizar testcontainers para testar a integração com o Weaviate
