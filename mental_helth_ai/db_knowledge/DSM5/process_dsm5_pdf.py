import json
import os

import nltk
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from rich import print

from .utils import reconstruct_documents, split_into_sentences

nltk.download('punkt')

DATA_PATH = 'data/'
FILE_NAME = 'DSM5_organized.pdf'
FULL_PATH = os.path.join(DATA_PATH, FILE_NAME)
TARGET_LINES_PER_CHUNK = 15

loader = PyPDFLoader(FULL_PATH)
documents = loader.load()

sentence_splits = [split_into_sentences(doc) for doc in documents]
sentence_splits = [
    sentence for sublist in sentence_splits for sentence in sublist
]

reconstructed_docs = reconstruct_documents(
    sentence_splits, target_lines_per_chunk=TARGET_LINES_PER_CHUNK
)

splitted_documents = [
    Document(
        page_content=chunk,
        metadata={
            'source': FILE_NAME,
            'start_index': idx * TARGET_LINES_PER_CHUNK,
        },
    )
    for idx, chunk in enumerate(reconstructed_docs)
]

splitted_documents = [
    doc for doc in splitted_documents if doc.page_content.strip()
]

splitted_docs_json = [doc.dict() for doc in splitted_documents]

with open(
    f'{DATA_PATH}/DSM5_splitted_documents.json', 'w', encoding='utf-8'
) as f:
    json.dump(splitted_docs_json, f, ensure_ascii=False)

print("Splitted documents saved to 'splitted_documents.json'")
