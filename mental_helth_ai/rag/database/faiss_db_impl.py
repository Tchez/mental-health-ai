import json
import os
from typing import Any, Dict, List, Tuple

import faiss
from rich import print
from sentence_transformers import SentenceTransformer

from mental_helth_ai.rag.database.db_interface import DatabaseInterface
from mental_helth_ai.settings import Settings

settings = Settings()


class FAISSDatabase(DatabaseInterface):
    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL,
        index_path: str = settings.FAISS_INDEX_PATH,
        embedding_dimension: int = settings.EMBEDDING_DIMENSION,
        documents_path: str = settings.DOCUMENTS_PATH,
    ):
        """Inicializa a instância do FAISSDatabase com os parâmetros fornecidos e tenta carregar o índice e documentos existentes."""  # noqa: E501
        self.model_name = model_name
        self.index_path = index_path
        self.embedding_dimension = embedding_dimension
        self.documents_path = documents_path
        self.model = SentenceTransformer(self.model_name)
        self.index = None
        self.documents = []

        try:
            self.load_index()
            self._load_documents()
            if self.documents:
                print(f'Loaded {len(self.documents)} documents')
            else:
                print('No documents loaded')
        except Exception as e:
            print(f'Error loading index or documents: {e}')

    def _save_documents(self):
        """Salva os documentos em um arquivo JSON, removendo os embeddings antes de salvar."""  # noqa: E501
        try:
            for doc in self.documents:
                if 'embedding' in doc:
                    del doc['embedding']
            with open(self.documents_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=4)
            print('Documents saved successfully')
        except Exception as e:
            raise RuntimeError(f'Failed to save documents: {e}')

    def _load_documents(self):
        """
        Carrega os documentos de um arquivo JSON.
        """
        try:
            if (
                os.path.exists(self.documents_path)
                and os.path.getsize(self.documents_path) > 0
            ):
                with open(self.documents_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                print('Documents loaded successfully')
            else:
                self.documents = []
                print(f'No documents to load from {self.documents_path}')
        except Exception as e:
            self.documents = []
            print(f'Failed to load documents: {e}')

    def load_index(self):
        """
        Carrega o índice FAISS a partir de um arquivo.
        """
        try:
            self.index = faiss.read_index(self.index_path)
            print('FAISS index loaded successfully')
            print(f'Index has {self.index.ntotal} embeddings')
        except Exception as e:
            print(f'Failed to load FAISS index, creating a new one: {e}')
            self.create_empty_index()

    def save_index(self):
        """
        Salva o índice FAISS em um arquivo.
        """
        try:
            faiss.write_index(self.index, self.index_path)
            print('FAISS index saved successfully')
        except Exception as e:
            raise RuntimeError(f'Failed to save FAISS index: {e}')

    def create_empty_index(self):
        """
        Cria um índice FAISS vazio com a dimensão de embedding especificada.
        """
        try:
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
            print('Empty FAISS index created successfully')
            print(f'Index has {self.index.ntotal} embeddings')
        except Exception as e:
            raise RuntimeError(f'Failed to create an empty FAISS index: {e}')

    def create_index(self, embeddings: List[List[float]]):
        """
        Cria um índice FAISS a partir dos embeddings fornecidos.
        """
        try:
            dimensions = len(embeddings[0])
            if dimensions != self.embedding_dimension:
                raise ValueError(
                    f'Embedding dimension mismatch: expected {
                        self.embedding_dimension
                    }, got {dimensions}'
                )
            self.index = faiss.IndexFlatL2(dimensions)
            self.index.add(embeddings)
            print('FAISS index created successfully with embeddings')
            print(f'Index has {self.index.ntotal} embeddings')
        except Exception as e:
            raise RuntimeError(f'Failed to create FAISS index: {e}')

    def add_documents(
        self, documents: List[Dict[str, Any]], batch_size: int = 500
    ):
        """
        Adiciona documentos ao índice FAISS em lotes.
        """
        try:
            total_documents = len(documents)
            for i in range(0, total_documents, batch_size):
                batch = documents[i : i + batch_size]
                texts = [doc['page_content'] for doc in batch]
                embeddings = self.model.encode(texts, show_progress_bar=True)

                dimensions = len(embeddings[0])
                if dimensions != self.embedding_dimension:
                    raise ValueError(
                        f'Embedding dimension mismatch: expected {
                            self.embedding_dimension
                        }, got {dimensions}'
                    )
                self.index.add(embeddings)
                print(
                    f'{min(i + batch_size, total_documents)} out of {
                        total_documents
                    } documents added successfully'
                )
            self.save_index()
            self.documents.extend(documents)
            self._save_documents()
        except AssertionError as ae:
            raise RuntimeError(f'Assertion error: {ae}')
        except ValueError as ve:
            raise RuntimeError(f'Value error: {ve}')
        except Exception as e:
            raise RuntimeError(f'Failed to add documents: {e}')

    def search(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Realiza uma busca no índice FAISS com base em uma query fornecida.
        """
        try:
            if self.index is None or self.index.ntotal == 0:
                raise RuntimeError('Index is not initialized or empty')
            query_embedding = self.model.encode([query])
            D, doc_index = self.index.search(query_embedding, top_k)

            results = [
                (self.documents[i], D[0][idx])
                for idx, i in enumerate(doc_index[0])
            ]
            return results
        except Exception as e:
            raise RuntimeError(f'Search failed: {e}')
