import json
import os
from typing import List

import nltk
from langchain_community.document_loaders import PyPDFLoader
from rich import print

from mental_helth_ai.processing_raw_data.utils import (
    reconstruct_documents,
    split_into_sentences,
)

nltk.download('punkt')

ARTICLES_PATH = 'data/raw/articles/'
OUTPUT_PATH = 'data/processed/articles/'
TARGET_LINES_PER_CHUNK = 15


def process_article(file_path: str) -> List[dict]:
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitted_documents = []
    for i, doc in enumerate(documents):
        page_number = i + 1
        sentence_splits = split_into_sentences(doc)
        reconstructed_docs = reconstruct_documents(
            sentence_splits, target_lines_per_chunk=TARGET_LINES_PER_CHUNK
        )
        for _, chunk in enumerate(reconstructed_docs):
            splitted_documents.append({
                'title': f'{os.path.splitext(os.path.basename(file_path))[0]} - Page {page_number}',  # noqa
                'page_content': chunk,
                'metadata': {
                    'type': 'article',
                    'source': file_path,
                    'page_number': page_number,
                    'source_description': f'Article from {os.path.basename(file_path)}',  # noqa
                    'date': '2018-10-02T00:00:00Z',
                },
            })

    return [doc for doc in splitted_documents if doc['page_content'].strip()]


def process_all_articles(directory: str):
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    for file_name in os.listdir(directory):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(directory, file_name)
            documents = process_article(file_path)

            json_file_path = os.path.join(
                OUTPUT_PATH, f'{os.path.splitext(file_name)[0]}.json'
            )
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(documents, f, ensure_ascii=False)

            print(
                f"Splitted documents for '{file_name}' saved to '{json_file_path}'"  # noqa
            )


process_all_articles(ARTICLES_PATH)
