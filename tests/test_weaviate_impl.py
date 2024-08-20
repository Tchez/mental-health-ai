from mental_helth_ai.rag.database.weaviate_impl import WeaviateClient


def test_db_init(weaviate_client: WeaviateClient):
    """Test that the WeaviateClient is initialized correctly."""
    assert weaviate_client is not None


def test_verify_db_connection(weaviate_client: WeaviateClient):
    """Test verifying the WeaviateClient connection."""
    assert weaviate_client.verify_database() is True


# def test_add_valid_document(weaviate_client: WeaviateClient, sample_document): # noqa E501
# TODO: descomentar quando instância do weaviate para teste for criada corretamente # noqa E501
#     """Test adding a valid document."""
#     result = weaviate_client.add_document(sample_document)
#     weaviate_client.delete_document_by_id(result.uuid)
#     assert result is True


# def test_get_document(weaviate_client: WeaviateClient, sample_document):
#     """Test retrieving a document."""
#     weaviate_client.add_document(sample_document)
#     result = weaviate_client.get_document_by_id(sample_document.id)
#     assert result.id == sample_document.id


# def test_delete_document(weaviate_client: WeaviateClient, sample_document):
#     """Test deleting a document."""
#     weaviate_client.add_document(sample_document)
#     result = weaviate_client.delete_document_by_id(sample_document.id)
#     assert result is True


# def test_add_invalid_document(
#     weaviate_client: WeaviateClient, invalid_document
# ):
#     """Test adding an invalid document."""
#     with pytest.raises(ValidationError):
#         weaviate_client.add_document(invalid_document)


# def test_validate_document(weaviate_client: WeaviateClient, sample_document):
#     """Test document validation."""
#     validated_document = weaviate_client._validate_document(sample_document)
#     assert validated_document.title == "Sample Document"


# def test_get_nonexistent_document(weaviate_client: WeaviateClient):
#     """Test retrieving a nonexistent document."""
#     result = weaviate_client.get_document_by_id("nonexistent-id")
#     assert result is None
