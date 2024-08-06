import json

from rich import print

from mental_helth_ai.rag.database.faiss_db_impl import FAISSDatabase
from mental_helth_ai.rag.embedding.sentence_transformer_impl import (
    SentenceTransformerEmbedding,
)

with open('data/dsm5/DSM5_splitted_documents.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)

embedding_generator = SentenceTransformerEmbedding()
db = FAISSDatabase()

embeddings = embedding_generator.generate_embeddings(documents)
print('Embeddings generated successfully')

for doc, embedding in zip(documents, embeddings):
    doc['embedding'] = embedding.tolist()

print('Embeddings added to documents successfully')

try:
    db.add_documents(documents)
    db.save_index()
    print('Documents added and index saved successfully')
except RuntimeError as e:
    print(f'Error: {e}')
