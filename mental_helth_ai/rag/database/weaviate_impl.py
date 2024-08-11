from http import HTTPStatus
from itertools import count
from typing import List

import weaviate
import weaviate.classes as wvc
from rich import print
from weaviate.collections.classes.types import WeaviateProperties
from weaviate.exceptions import UnexpectedStatusCodeError

from mental_helth_ai.rag.database.db_interface import DatabaseInterface
from mental_helth_ai.rag.database.utils import read_json_in_nested_path
from mental_helth_ai.settings import settings


class WeaviateImpl(DatabaseInterface):
    def __init__(
        self,
        host=settings.WEAVIATE_URL,
        port=settings.WEAVIATE_PORT,
        insert_batch_size=100,
        insert_max_attempts=3,
    ):
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

    def get_session(self):
        """Get a session to interact with the database."""
        with self.client as client:
            yield client

    def init_db(self) -> None:
        """Initialize the database with the necessary classes and properties.
        If the class already exists, it will not be created again."""
        with self.client as client:
            try:
                print("Creating class 'Documents'...")
                client.collections.create(
                    name='Documents',
                    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers(),
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
                    print('Collection already exists in the database.')
            except Exception as e:
                print(f'Failed to create collection: {e}')

    def delete_db_collections(self) -> None:
        """Delete all collections in the database."""
        with self.client as client:
            try:
                print('Deleting all collections...')
                collections = client.collections.list_all()
                if len(collections) == 0:
                    print('No collections found.')
                    return
                for collection in collections:
                    print(f'Deleting collection {collection}...')
                    client.collections.delete(collection)
                    print(f'Collection {collection} deleted.')
            except Exception as e:
                print(f'Failed to delete collections: {e}')

    def get_db_info(self) -> None:
        """Get information about the database."""
        with self.client as client:
            try:
                document_collection = client.collections.get('Documents')
                if not document_collection.exists():
                    print("Collection 'Documents' not found.")
                    return

                aggregation_document = document_collection.aggregate.over_all(
                    total_count=True
                )

                if aggregation_document.total_count == 0:
                    print('Collection is empty.')
                    return

                print(
                    f'Exemplo de documento:\n{next(document_collection.iterator())}'  # noqa: E501
                )
                print(
                    f'Total de documentos: {aggregation_document.total_count}'
                )

            except Exception as e:
                print(f'Failed to get database information: {e}')

    def load_documents(self, root_path: str) -> bool:
        """Load documents to the database from a directory and its subdirectories.

        Args:
            root_path (str): Path to the root directory where the documents are located.
        """  # noqa: E501

        json_files = read_json_in_nested_path(root_path)

        if len(json_files) == 0:
            print('No documents found.')
            return

        print(f'Adding {len(json_files)} files to the database...')

        def batch_insert_documents(documents: List[dict]):
            for i in range(0, len(documents), self.insert_batch_size):
                yield documents[i : i + self.insert_batch_size]

        total_counter = count(start=1)

        for batch in batch_insert_documents(json_files):  # noqa: PLR1702
            attempts = 0
            success = False

            while attempts < self.insert_max_attempts and not success:
                with self.client as client:
                    try:
                        for documents in batch:
                            print(f'Adding {len(documents)} documents...')
                            for doc in documents:
                                doc_number = next(total_counter)
                                print(f'Adding document {doc_number}...')

                                document_collection = client.collections.get(
                                    'Documents'
                                )

                                uuid = document_collection.data.insert({
                                    'title': doc.get('title', ''),
                                    'page_content': doc.get(
                                        'page_content', ''
                                    ),
                                    'metadata': doc.get('metadata', {}),
                                })

                                print(f'Document added with UUID: {uuid}')

                                success = True
                    except Exception as e:
                        print(f'Failed to insert document: {e}')
                        attempts += 1

            if not success:
                print(
                    f'Failed to insert batch of documents after {attempts} attempts.'  # noqa: E501
                )
                return False

        return True

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
                    print("Collection 'Documents' not found.")
                    return []

                search_result = document_collection.query.near_text(
                    query=query,
                    limit=limit,
                    return_metadata=wvc.query.MetadataQuery(
                        distance=True, score=True
                    ),
                )

                if len(search_result.objects) == 0:
                    print('No documents found.')
                    return []

                return search_result.objects
            except Exception as e:
                print(f'Failed to search documents: {e}')
                return []
