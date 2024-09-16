import pytest
from pydantic import ValidationError

from mental_health_ai.rag.database.schemas import Metadata, WeaviateDocument


def test_valid_document(sample_document):
    """Test creating a valid WeaviateDocument."""
    document = WeaviateDocument(**sample_document)
    assert document.title == 'Sample Document'


def test_invalid_document(invalid_document):
    """Test that creating an invalid WeaviateDocument raises an error."""
    with pytest.raises(ValidationError):
        WeaviateDocument(**invalid_document)


def test_metadata_validation():
    """Test validation of the Metadata schema."""
    metadata = Metadata(
        type='article',
        source='https://example.com',
        page_number=1,
        source_description='Example site',
        date='2023-01-01T00:00:00Z',
    )
    assert metadata.type == 'article'


def test_metadata_invalid_type():
    """Test that an empty 'type' field in Metadata raises an error."""
    with pytest.raises(ValidationError):
        Metadata(
            type='',
            source='https://example.com',
            page_number=1,
            source_description='Example site',
            date='2023-01-01T00:00:00Z',
        )
