from typing import List
from langchain.schema import Document
import json

with open('data/cid/transtornos_mentais_e_comportamentais.json', 'r') as f:
    cid_list = json.load(f)


def create_cid_documents(cid_list: List[dict]) -> List[Document]:
    documents = []

    for item in cid_list:
        cid_code, illness, details = (
            item['codigo'],
            item['doenca'],
            item['detalhes'],
        )

        page_content = f"# {illness}\n\nDetalhes: {details}"
        document = Document(
            page_content=page_content,
            metadata={'code': cid_code, 'type': 'cid'},
        )

        documents.append(document)
    return documents
