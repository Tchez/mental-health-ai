import json
import os
from typing import List

import nltk
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from rich import print

from mental_helth_ai.db_knowledge.utils import (
    reconstruct_documents,
    split_into_sentences,
)

nltk.download('punkt')

ARTICLES_PATH = 'data/articles/original/'
TARGET_LINES_PER_CHUNK = 15


def process_article(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitted_documents = []
    for i, doc in enumerate(documents):
        page_number = i + 1
        sentence_splits = split_into_sentences(doc)
        reconstructed_docs = reconstruct_documents(
            sentence_splits, target_lines_per_chunk=TARGET_LINES_PER_CHUNK
        )
        for idx, chunk in enumerate(reconstructed_docs):
            splitted_documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        'source': os.path.basename(file_path),
                        'start_index': idx * TARGET_LINES_PER_CHUNK,
                        'page_number': page_number,
                    },
                )
            )
    return [doc for doc in splitted_documents if doc.page_content.strip()]


def process_all_articles(directory: str):
    for file_name in os.listdir(directory):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(directory, file_name)
            documents = process_article(file_path)
            splitted_docs_json = [doc.dict() for doc in documents]

            json_file_path = os.path.join(
                directory, f'{os.path.splitext(file_name)[0]}.json'
            )
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(splitted_docs_json, f, ensure_ascii=False)

            print(
                f"Splitted documents for '{file_name}' saved to '{json_file_path}'"
            )


process_all_articles(ARTICLES_PATH)
