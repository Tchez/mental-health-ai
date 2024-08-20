from http import HTTPStatus
from itertools import count
from typing import Any, Dict, List, Union

import weaviate
import weaviate.classes as wvc
from pydantic import ValidationError
from rich import print
from weaviate.collections.classes.internal import ObjectSingleReturn
from weaviate.collections.classes.types import WeaviateProperties
from weaviate.exceptions import UnexpectedStatusCodeError

from mental_helth_ai.rag.database.db_interface import DatabaseInterface
from mental_helth_ai.rag.database.schemas import DataModel, WeaviateDocument
from mental_helth_ai.rag.database.utils import read_json_in_nested_path
from mental_helth_ai.settings import settings


class WeaviateClient(DatabaseInterface):
    def __init__(
        self,
        local_embeddings=settings.IS_LOCAL_EMBEDDING,
        host=settings.WEAVIATE_URL,
        port=settings.WEAVIATE_PORT,
        insert_batch_size=100,
        insert_max_attempts=3,
    ):
        self.local_embeddings = local_embeddings
        self.insert_batch_size = insert_batch_size
        self.insert_max_attempts = insert_max_attempts
        self.client = weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=50051,
            additional_config=wvc.init.AdditionalConfig(
                timeout=wvc.init.Timeout(init=30, query=300, insert=400)
            ),
        )

    @staticmethod
    def _handle_exception(e: Exception, message: str):
        """Handle exceptions and log the error message."""
        print(f'[red]{message}: {e}[/red]')
        raise e

    @staticmethod
    def _validate_document(document: Dict[str, Any]) -> WeaviateDocument:
        """Validate a single document using Pydantic schema."""
        try:
            return WeaviateDocument(**document)
        except ValidationError as e:
            print(f'[red]Validation failed for document: {e}[/red]')
            raise e

    def _validate_documents(
        self,
        documents: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        continue_on_error: bool = False,
    ) -> DataModel:
        """Validate a batch of documents using Pydantic schema."""
        validated_documents = []

        if isinstance(documents[0], list):
            documents = [item for sublist in documents for item in sublist]

        for doc in documents:
            try:
                validated_documents.append(self._validate_document(doc))
            except ValidationError as e:
                print(f'[red]Validation failed for document: {e}[/red]')
                if not continue_on_error:
                    break

        if not validated_documents:
            raise ValidationError('No valid documents found.')
        return DataModel(documents=validated_documents)

    def get_session(self):
        """Get a session to interact with the database."""
        with self.client as client:
            yield client

    def verify_database(self) -> bool:
        """Verify if the database is up and running."""
        with self.client as client:
            try:
                print('Verifying database...')
                if not client:
                    raise RuntimeError('Database is not connected.')

                if not client.is_connected():
                    raise RuntimeError('Database is not connected.')

                if not client.is_ready():
                    raise RuntimeError('Database is not ready.')

                if not client.is_live():
                    raise RuntimeError('Database is not live.')

                document_collection = client.collections.get('Documents')

                if not document_collection.exists():
                    raise RuntimeError('Collection Documents not found.')

                aggregation_document = document_collection.aggregate.over_all(
                    total_count=True
                )

                if aggregation_document.total_count == 0:
                    raise RuntimeError('Collection is empty.')

                print('[green]Database is up and running.[/green]')
                return True
            except Exception as e:
                self._handle_exception(e, 'Failed to verify database')
                return False

    def initialize_database(self) -> None:
        """Initialize the database with the necessary classes and properties.
        If the class already exists, it will not be created again."""
        with self.client as client:
            try:
                print("Creating class 'Documents'...")
                client.collections.create(
                    name='Documents',
                    vectorizer_config=(
                        wvc.config.Configure.Vectorizer.text2vec_transformers()
                        if self.local_embeddings
                        else wvc.config.Configure.Vectorizer.text2vec_openai()
                    ),
                    properties=[
                        wvc.config.Property(
                            name='title',
                            description='Title of the document (e.g., article ttle)',  # noqa: E501
                            data_type=wvc.config.DataType.TEXT,
                        ),
                        wvc.config.Property(
                            name='page_content',
                            description='The content of the document',
                            data_type=wvc.config.DataType.TEXT,
                        ),
                        wvc.config.Property(
                            name='metadata',
                            description='Metadata of the document',
                            data_type=wvc.config.DataType.OBJECT,
                            nested_properties=[
                                wvc.config.Property(
                                    name='type',
                                    description='Type of the document (e.g., article, blog post)',  # noqa: E501
                                    data_type=wvc.config.DataType.TEXT,
                                ),
                                wvc.config.Property(
                                    name='source',
                                    data_type=wvc.config.DataType.TEXT,
                                    description='The source of the document (e.g., URL, site name)',  # noqa: E501
                                ),
                                wvc.config.Property(
                                    name='page_number',
                                    description='The page number of the document',  # noqa: E501
                                    data_type=wvc.config.DataType.NUMBER,
                                ),
                                wvc.config.Property(
                                    name='source_description',
                                    description='Description of the source',
                                    data_type=wvc.config.DataType.TEXT,
                                ),
                                wvc.config.Property(
                                    name='date',
                                    description='The publication date of the document',  # noqa: E501
                                    data_type=wvc.config.DataType.DATE,
                                ),
                            ],
                        ),
                    ],
                )
            except UnexpectedStatusCodeError as e:
                if e.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
                    if 'already exists' in e.message:
                        print(
                            '[yellow]Collection already exists in the database.[/yellow]'  # noqa: E501 #TODO: ALterar para printar apenas se already in message # noqa: E501
                        )
                    else:
                        self._handle_exception(
                            e, 'Failed to create collection'
                        )
                else:
                    self._handle_exception(e, 'Failed to create collection')
            except Exception as e:
                self._handle_exception(e, 'Failed to create collection')

    def delete_all_collections(self) -> None:
        """Delete all collections in the database."""
        with self.client as client:
            try:
                print('Deleting all collections...')
                collections = client.collections.list_all()
                if len(collections) == 0:
                    print('[yellow]No collections found.[/yellow]')
                    return
                for collection in collections:
                    print(f'Deleting collection {collection}...')
                    client.collections.delete(collection)
                    print(f'Collection {collection} deleted.')
            except Exception as e:
                self._handle_exception(e, 'Failed to delete collections')

    def get_database_info(self) -> None:
        """Get information about the database."""
        with self.client as client:
            try:
                document_collection = client.collections.get('Documents')
                if not document_collection.exists():
                    print("[yellow]Collection 'Documents' not found.[/yellow]")
                    return

                aggregation_document = document_collection.aggregate.over_all(
                    total_count=True
                )

                if aggregation_document.total_count == 0:
                    print('[yellow]Collection is empty.[/yellow]')
                    return

                print(
                    f'Example document:\n{next(document_collection.iterator())}'  # noqa: E501
                )
                print(f'Total documents: {aggregation_document.total_count}')

            except Exception as e:
                self._handle_exception(e, 'Failed to get database information')

    def _batch_insert_documents(self, documents: List[Dict[str, Any]]):
        """Helper function to insert documents in batches."""
        total_counter = count(start=1)
        for batch in self._split_into_batches(documents):
            attempts = 0
            success = False
            while attempts < self.insert_max_attempts and not success:
                with self.client as client:
                    try:
                        print(f'Inserting {len(batch)} documents...')
                        for doc in batch:
                            doc_number = next(total_counter)
                            print(f'Inserting document {doc_number}...')
                            document_collection = client.collections.get(
                                'Documents'
                            )
                            uuid = document_collection.data.insert({
                                'title': doc.get('title', ''),
                                'page_content': doc.get('page_content', ''),
                                'metadata': doc.get('metadata', {}),
                            })
                            print(f'Document added with UUID: {uuid}')
                        success = True
                    except Exception as e:
                        self._handle_exception(e, 'Failed to insert document')
                        attempts += 1
            if not success:
                print(
                    f'Failed to insert batch of documents after {attempts} attempts.'  # noqa: E501
                )
                raise RuntimeError(
                    'Batch insert failed after multiple attempts'
                )

    def _split_into_batches(self, documents: List[dict]):
        """Split documents into batches."""
        for i in range(0, len(documents), self.insert_batch_size):
            yield documents[i : i + self.insert_batch_size]

    def load_documents(
        self, root_path: str, continue_on_error: bool = False
    ) -> bool:
        """Load documents to the database from a directory and its subdirectories."""  # noqa: E501

        json_files = read_json_in_nested_path(root_path)
        if len(json_files) == 0:
            print('[yellow]No documents found.[/yellow]')
            return False

        print(f'Validating {len(json_files)} files...')

        try:
            validated_data = self._validate_documents(
                json_files, continue_on_error
            )
            print(
                f'Adding {len(validated_data.documents)} validated documents to the database...'  # noqa: E501
            )

            self._batch_insert_documents([
                doc.model_dump() for doc in validated_data.documents
            ])
            return True
        except ValidationError as e:
            self._handle_exception(e, 'Failed to validate documents')
            return False
        except Exception as e:
            self._handle_exception(e, 'Failed to load documents')
            return False

    def search(self, query: str, limit: int = 5) -> List[WeaviateProperties]:
        """Search for documents in the database using a query.

        Args:
            query (str): Query to search for documents.
            limit (int): Maximum number of documents to return.

        Returns:
            List[dict]: List of documents that match the query.
        """
        with self.client as client:
            try:
                document_collection = client.collections.get('Documents')

                if not document_collection.exists():
                    print("[yellow]Collection 'Documents' not found.[/yellow]")
                    return []

                search_result = document_collection.query.near_text(
                    query=query,
                    limit=limit,
                    return_metadata=wvc.query.MetadataQuery(
                        distance=True, score=True
                    ),
                )

                if len(search_result.objects) == 0:
                    print('[yellow]No documents found.[/yellow]')
                    return []

                return search_result.objects
            except Exception as e:
                self._handle_exception(e, 'Failed to search documents')
                return []

    def get_document_by_id(
        self, document_id: str
    ) -> ObjectSingleReturn | None:
        """Get a document by its ID.

        Args:
            document_id (str): UUID of the document.

        Returns:
            Dict[str, Any]: Document with the given ID.
        """
        with self.client as client:
            try:
                document_collection = client.collections.get('Documents')

                if not document_collection.exists():
                    print("[yellow]Collection 'Documents' not found.[/yellow]")
                    return None

                document = document_collection.query.fetch_object_by_id(
                    document_id
                )

                if not document:
                    print(
                        f'[yellow]Document with ID {document_id} not found.[/yellow]'  # noqa: E501
                    )
                    return None

                return document
            except ValueError as e:
                print(
                    f'[red]Document with ID {document_id} not found: {e}[/red]'
                )
                return None
            except Exception as e:
                self._handle_exception(e, 'Failed to get document')
                return None

    def add_document(self, document: Dict[str, Any]) -> bool:
        """Add a document to the database.

        Args:
            document (Dict[str, Any]): Document to be added to the database.

        Examples:
            >>> document = {
            ...     'title': 'Document title',
            ...     'page_content': 'Document content',
            ...     'metadata': {
            ...         'type': 'article',
            ...         'source': 'https://example.com',
            ...         'page_number': 1,
            ...         'source_description': 'Example site',
            ...         'date': '2021-10-01T00:00:00Z',
            ...     },
            ... }
            >>> add_document(document)
            Document added with UUID: 12345678-1234-1234-1234-1234567890a
        """  # noqa: E501

        validated_document = self._validate_document(document)

        with self.client as client:
            try:
                document_collection = client.collections.get('Documents')

                if not document_collection.exists():
                    print("[yellow]Collection 'Documents' not found.[/yellow]")
                    return False

                uuid = document_collection.data.insert(
                    validated_document.model_dump()
                )

                print(f'Document added with UUID: {uuid}')
                return True
            except Exception as e:
                self._handle_exception(
                    e, 'Failed to add document to the database'
                )
                return False

    def delete_document_by_id(self, document_id: str) -> None:
        """Delete a document by its UUID.

        Args:
            document_id (str): UUID of the document to be deleted.
        """  # noqa: E501
        with self.client as client:
            try:
                document_collection = client.collections.get('Documents')

                if not document_collection.exists():
                    print("[yellow]Collection 'Documents' not found.[/yellow]")
                    return

                document_collection.data.delete_by_id(document_id)

                print(f'Document with ID {document_id} deleted.')
            except ValueError as e:
                print(
                    f'[red]Document with ID {document_id} not found: {e}[/red]'
                )
            except Exception as e:
                self._handle_exception(e, 'Failed to delete document')
