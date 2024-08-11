import json
import os

import nltk
from langchain_community.document_loaders import PyPDFLoader
from rich import print

from mental_helth_ai.processing_raw_data.utils import (
    reconstruct_documents,
    split_into_sentences,
)

nltk.download('punkt')

RAW_DATA_PATH = 'data/raw/'
OUTPUT_PATH = 'data/processed/'
FILE_NAME = 'DSM5_organized.pdf'
FULL_PATH = os.path.join(RAW_DATA_PATH, FILE_NAME)
TARGET_LINES_PER_CHUNK = 15

loader = PyPDFLoader(FULL_PATH)
documents = loader.load()

splitted_documents = []

for i, doc in enumerate(documents):
    page_number = i + 1
    sentence_splits = split_into_sentences(doc)
    reconstructed_docs = reconstruct_documents(
        sentence_splits, target_lines_per_chunk=TARGET_LINES_PER_CHUNK
    )
    for idx, chunk in enumerate(reconstructed_docs):
        splitted_documents.append({
            'title': f'DSM-5 Page {page_number}',
            'page_content': chunk,
            'metadata': {
                'type': 'DSM-5',
                'source': FILE_NAME,
                'page_number': page_number,
                'source_description': 'O Manual Diagnóstico e Estatístico de Transtornos Mentais 5.ª edição, ou DSM-5, é um manual diagnóstico e estatístico feito pela Associação Americana de Psiquiatria para definir como é feito o diagnóstico de transtornos mentais. Usado por psicólogos, fonoaudiólogos, médicos e terapeutas ocupacionais. A versão atualizada saiu em maio de 2013 e substitui o DSM-IV criado em 1994 e revisado em 2000. Desde o DSM-I, criado em 1952, esse manual tem sido uma das bases de diagnósticos de saúde mental mais usados no mundo.',  # noqa
                'date': '2013-05-18T00:00:00Z',
            },
        })

splitted_documents = [
    doc for doc in splitted_documents if doc['page_content'].strip()
]

with open(f'{OUTPUT_PATH}dsm5.json', 'w', encoding='utf-8') as f:
    json.dump(splitted_documents, f, ensure_ascii=False)

print('Splitted documents saved as JSON file')
