from http import HTTPStatus
from itertools import count
from typing import Any, Dict, Generator, List, Optional, Union

import weaviate
import weaviate.classes as wvc
from pydantic import ValidationError
from rich import print
from weaviate.collections.classes.internal import ObjectSingleReturn
from weaviate.collections.classes.types import WeaviateProperties
from weaviate.exceptions import UnexpectedStatusCodeError

from mental_health_ai.rag.database.db_interface import DatabaseInterface
from mental_health_ai.rag.database.schemas import DataModel, WeaviateDocument
from mental_health_ai.rag.database.utils import read_json_in_nested_path
from mental_health_ai.settings import settings


class WeaviateClient(DatabaseInterface):
    """
    A client to interact with a Weaviate vector database.

    This class provides methods to initialize the database, insert documents,
    perform searches, and manage collections. It also implements a context manager
    to handle the connection lifecycle.

    Attributes:
        local_embeddings (bool): Whether to use local embeddings or a remote vectorizer.
        insert_batch_size (int): Number of documents to insert in a single batch (default is 100).
        insert_max_attempts (int): Maximum number of attempts to insert a batch (default is 3).
    """  # noqa: E501

    def __init__(
        self,
        local_embeddings: bool = settings.IS_LOCAL_EMBEDDING,
        host: str = settings.WEAVIATE_URL,
        port: int = settings.WEAVIATE_PORT,
        insert_batch_size: int = 100,
        insert_max_attempts: int = 3,
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

    def __enter__(self):
        """Enter the runtime context related to this object."""
        if not self.client.is_connected():
            self.client.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context and close the client connection."""
        if self.client.is_connected():
            self.client.close()

    @staticmethod
    def _handle_exception(e: Exception, message: str):
        """Handle exceptions and log the error message."""
        print(f'[red]{message}: {e}[/red]')
        raise e

    @staticmethod
    def _validate_document(document: Dict[str, Any]) -> WeaviateDocument:
        """Validate a single document using the Pydantic schema."""
        try:
            return WeaviateDocument(**document)
        except ValidationError as e:
            print(f'[red]Validation failed for document: {e}[/red]')
            print(f'[red]Document: {document}[/red]')
            raise e

    def _validate_documents(
        self,
        documents: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        continue_on_error: bool = False,
    ) -> DataModel:
        """
        Validate a batch of documents using the Pydantic schema.

        Args:
            documents (Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]): Documents to validate.
            continue_on_error (bool): Whether to continue validating after an error.

        Returns:
            DataModel: A DataModel containing validated documents.

        Raises:
            ValidationError: If no valid documents are found.
        """  # noqa: E501
        validated_documents = []

        if documents and isinstance(documents[0], list):
            documents = [item for sublist in documents for item in sublist]

        for doc in documents:
            try:
                validated_documents.append(self._validate_document(doc))
            except ValidationError as e:
                print(f'[red]Validation failed for document: {e}[/red]')
                if not continue_on_error:
                    raise e

        if not validated_documents:
            raise ValidationError('No valid documents found.')
        return DataModel(documents=validated_documents)

    def verify_database(self) -> bool:
        """
        Verify if the database is up and running.

        Returns:
            bool: True if the database is operational, False otherwise.
        """
        try:
            print('Verifying database...')
            if not self.client.is_ready():
                raise RuntimeError('Database is not ready.')

            if not self.client.is_live():
                raise RuntimeError('Database is not live.')

            document_collection = self.client.collections.get('Documents')

            if not document_collection.exists():
                raise RuntimeError("Collection 'Documents' not found.")

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
        """
        Initialize the database with the necessary classes and properties.

        If the class already exists, it will not be created again.
        """
        try:
            print("Creating class 'Documents'...")
            self.client.collections.create(
                name='Documents',
                vectorizer_config=(
                    wvc.config.Configure.Vectorizer.text2vec_transformers()
                    if self.local_embeddings
                    else wvc.config.Configure.Vectorizer.text2vec_openai()
                ),
                properties=[
                    wvc.config.Property(
                        name='title',
                        description='Title of the document (e.g., article title)',  # noqa: E501
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
                                description='The page number of the document',
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
            print("[green]Class 'Documents' created successfully.[/green]")
        except UnexpectedStatusCodeError as e:
            if (
                e.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
                and 'already exists' in e.message
            ):
                print(
                    '[yellow]Collection already exists in the database.[/yellow]'  # noqa: E501
                )
            else:
                self._handle_exception(e, 'Failed to create collection')
        except Exception as e:
            self._handle_exception(e, 'Failed to create collection')

    def delete_all_collections(self) -> None:
        """
        Delete all collections in the database.
        """
        try:
            print('Deleting all collections...')
            collections = self.client.collections.list_all()
            if not collections:
                print('[yellow]No collections found.[/yellow]')
                return
            for collection in collections:
                print(f"Deleting collection '{collection}'...")
                self.client.collections.delete(collection)
                print(f"Collection '{collection}' deleted.")
            print('[green]All collections deleted successfully.[/green]')
        except Exception as e:
            self._handle_exception(e, 'Failed to delete collections')

    def get_database_info(self) -> None:
        """
        Get information about the database, such as total documents and an example document.
        """  # noqa: E501
        try:
            document_collection = self.client.collections.get('Documents')
            if not document_collection.exists():
                print("[yellow]Collection 'Documents' not found.[/yellow]")
                return

            aggregation_document = document_collection.aggregate.over_all(
                total_count=True
            )

            if aggregation_document.total_count == 0:
                print('[yellow]Collection is empty.[/yellow]')
                return

            example_doc = next(document_collection.iterator(), None)
            if example_doc:
                print(f'Example document:\n{example_doc}')
            print(f'Total documents: {aggregation_document.total_count}')

        except Exception as e:
            self._handle_exception(e, 'Failed to get database information')

    def _batch_insert_documents(self, documents: List[Dict[str, Any]]):
        """
        Helper function to insert documents in batches.

        Args:
            documents (List[Dict[str, Any]]): Documents to insert.
        """
        total_counter = count(start=1)
        for batch in self._split_into_batches(documents):
            attempts = 0
            success = False
            while attempts < self.insert_max_attempts and not success:
                try:
                    print(f'Inserting {len(batch)} documents...')
                    document_collection = self.client.collections.get(
                        'Documents'
                    )
                    if not document_collection.exists():
                        raise RuntimeError("Collection 'Documents' not found.")

                    for doc in batch:
                        doc_number = next(total_counter)
                        print(f'Inserting document {doc_number}...')
                        uuid = document_collection.data.insert({
                            'title': doc.get('title', ''),
                            'page_content': doc.get('page_content', ''),
                            'metadata': doc.get('metadata', {}),
                        })
                        print(f'Document added with UUID: {uuid}')
                    success = True
                except Exception as e:
                    attempts += 1
                    print(f'[red]Attempt {attempts} failed: {e}[/red]')
                    if attempts >= self.insert_max_attempts:
                        self._handle_exception(e, 'Failed to insert documents')
            if not success:
                print(
                    f'Failed to insert batch of documents after {attempts} attempts.'  # noqa: E501
                )
                raise RuntimeError(
                    'Batch insert failed after multiple attempts'
                )

    def _split_into_batches(
        self, documents: List[Dict[str, Any]]
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Split documents into batches.

        Args:
            documents (List[Dict[str, Any]]): Documents to split.

        Returns:
            Generator[List[Dict[str, Any]], None, None]: Generator yielding document batches.
        """  # noqa: E501
        for i in range(0, len(documents), self.insert_batch_size):
            yield documents[i : i + self.insert_batch_size]

    def load_documents(
        self, root_path: str, continue_on_error: bool = False
    ) -> bool:
        """
        Load documents into the database from JSON files in a directory.

        Args:
            root_path (str): The root directory to search for JSON files.
            continue_on_error (bool): Whether to continue loading after an error.

        Returns:
            bool: True if documents were loaded successfully, False otherwise.
        """  # noqa: E501
        json_files = read_json_in_nested_path(root_path)
        if not json_files:
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
            print('[green]Documents loaded successfully.[/green]')
            return True
        except ValidationError as e:
            self._handle_exception(e, 'Failed to validate documents')
            return False
        except Exception as e:
            self._handle_exception(e, 'Failed to load documents')
            return False

    def search(self, query: str, limit: int = 5) -> List[WeaviateProperties]:
        """
        Search for documents in the database using a query.

        Args:
            query (str): The query string.
            limit (int): Maximum number of documents to return.

        Returns:
            List[WeaviateProperties]: List of documents that match the query.
        """
        try:
            document_collection = self.client.collections.get('Documents')

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

            if not search_result.objects:
                print('[yellow]No documents found.[/yellow]')
                return []

            return search_result.objects
        except Exception as e:
            self._handle_exception(e, 'Failed to search documents')
            return []

    def get_document_by_id(
        self, document_id: str
    ) -> Optional[ObjectSingleReturn]:
        """
        Get a document by its UUID.

        Args:
            document_id (str): UUID of the document.

        Returns:
            Optional[ObjectSingleReturn]: The document if found, else None.
        """
        try:
            document_collection = self.client.collections.get('Documents')

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
            print(f'[red]Invalid document ID {document_id}: {e}[/red]')
            return None
        except Exception as e:
            self._handle_exception(e, 'Failed to get document')
            return None

    def add_document(self, document: Dict[str, Any]) -> bool:
        """
        Add a single document to the database.

        Args:
            document (Dict[str, Any]): The document data.

        Returns:
            bool: True if the document was added successfully, False otherwise.

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
            >>> client.add_document(document)
        """
        validated_document = self._validate_document(document)

        try:
            document_collection = self.client.collections.get('Documents')

            if not document_collection.exists():
                print("[yellow]Collection 'Documents' not found.[/yellow]")
                return False

            uuid = document_collection.data.insert(
                validated_document.model_dump()
            )
            print(f'Document added with UUID: {uuid}')
            return True
        except Exception as e:
            self._handle_exception(e, 'Failed to add document to the database')
            return False

    def delete_document_by_id(self, document_id: str) -> None:
        """
        Delete a document by its UUID.

        Args:
            document_id (str): UUID of the document to be deleted.
        """
        try:
            document_collection = self.client.collections.get('Documents')

            if not document_collection.exists():
                print("[yellow]Collection 'Documents' not found.[/yellow]")
                return

            document_collection.data.delete_by_id(document_id)
            print(f'Document with ID {document_id} deleted.')
        except ValueError as e:
            print(f'[red]Invalid document ID {document_id}: {e}[/red]')
        except Exception as e:
            self._handle_exception(e, 'Failed to delete document')

    def get_documents_by_type_and_page_number(
        self, doc_type: str, page_number: int, source: Optional[str] = None
    ) -> List[WeaviateProperties]:
        """
        Get documents by type, page number, and optionally source.

        Args:
            doc_type (str): The type of the document (e.g., 'article', 'dsm-5').
            page_number (int): The page number.
            source (str, optional): The source of the document.

        Returns:
            List[WeaviateProperties]: List of documents matching the criteria.
        """  # noqa: E501
        try:
            document_collection = self.client.collections.get('Documents')
            if not document_collection.exists():
                print("[red]Collection 'Documents' not found.[/red]")
                return []

            documents: List[WeaviateProperties] = []

            for doc in document_collection.iterator():
                metadata = doc.properties.get('metadata', {})
                if (
                    metadata.get('type', '').lower() == doc_type.lower()
                    and int(metadata.get('page_number', -1)) == page_number
                    and (
                        source is None or metadata.get('source', '') == source
                    )
                ):
                    print(
                        f"Found document: {doc.properties.get('title', 'No Title')}"  # noqa: E501
                    )
                    documents.append(doc)

            if not documents:
                print(
                    '[yellow]No documents found matching the criteria.[/yellow]'  # noqa: E501
                )
            return documents
        except Exception as e:
            self._handle_exception(
                e, 'Failed to get documents by type and page number'
            )
            return []
