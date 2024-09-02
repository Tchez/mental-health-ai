import json
import os
from itertools import count

from langchain_community.document_loaders import PyPDFLoader
from rich import print

from mental_helth_ai.processing_raw_data.utils import (
    reconstruct_documents,
    split_into_sentences,
)

RAW_DATA_PATH = 'data/raw/articles/scrapped/'
OUTPUT_PATH = 'data/processed/articles/scrapped/'
METADATA_PATH = 'data/raw/articles/articles_metadata.json'
TARGET_LINES_PER_CHUNK = 15
LIMIT_TITLE_LENGTH = 200

with open(METADATA_PATH, 'r', encoding='utf-8') as meta_file:
    metadata_list = json.load(meta_file)[:100]
    cleaned_titles = [
        (
            f'{article["title"][:LIMIT_TITLE_LENGTH]}...'
            if len(article["title"]) > LIMIT_TITLE_LENGTH
            else article["title"]
        )
        .lower()
        .replace('\n', '')
        .replace(' ', '_')
        .replace('/', '_')
        .replace(',', '')
        + '.pdf'
        for article in metadata_list
    ]


def find_metadata_by_pdf(pdf_name, cleaned_titles):
    for indice, cleaned_title in enumerate(cleaned_titles):
        if pdf_name == cleaned_title:
            return metadata_list[indice]
    print(f'[red]Metadata not found for {pdf_name}[/red]')
    return None


pdf_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.pdf')]
counter = count(1)

for pdf_file in pdf_files:
    file_path = os.path.join(RAW_DATA_PATH, pdf_file)
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    metadata = find_metadata_by_pdf(pdf_file, cleaned_titles)

    if metadata:
        splitted_documents = []
        for i, doc in enumerate(documents):
            page_number = i + 1
            sentence_splits = split_into_sentences(doc)
            reconstructed_docs = reconstruct_documents(
                sentence_splits, target_lines_per_chunk=TARGET_LINES_PER_CHUNK
            )
            for idx, chunk in enumerate(reconstructed_docs):
                if not chunk.strip():
                    continue

                splitted_documents.append(
                    {
                        'title': f'{metadata["title"]} - Page {page_number}',
                        'page_content': chunk,
                        'metadata': {
                            'type': 'Article',
                            'source': pdf_file,
                            'page_number': page_number,
                            'source_description': metadata['description'],
                            'date': metadata['date'],
                        },
                    }
                )

        output_file_path = os.path.join(
            OUTPUT_PATH, f'{os.path.splitext(pdf_file)[0]}.json'
        )
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(splitted_documents, f, ensure_ascii=False)

        print(
            f'{next(counter)} - Documents for {pdf_file} saved as JSON at {output_file_path}'
        )
    else:
        print(f'{next(counter)} - No metadata found for {pdf_file}')
