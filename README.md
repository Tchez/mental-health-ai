# Projeto Mental Health AI

## Instalação

### Requisitos

- Python 3.12.*
- Poetry

### Passos

1. Clone o repositório:
    ```sh
    git clone https://github.com/Tchez/mental-health-ai.git
    cd mental-health-ai
    ```

2. Instale as dependências do projeto:
    ```sh
    poetry install
    ```

3. Configure as variáveis de ambiente:
    Crie um arquivo `.env` na raiz do projeto com base no arquivo `.env.example` e altere os valores conforme necessário.

4. Instale faiss-cpu:
    ```sh
    pip install faiss-cpu
    ```

### Uso

1. Ative o ambiente virtual:
    ```sh
    poetry shell
    ```

2. Rode o arquivo da implementação do banco de dados de maneira interativa:
    ```sh
    python -i mental_helth_ai/rag/database/faiss_db_impl.py
    ```

3. Crie uma instância do banco de dados:
    > Lembre-se de colocar o caminho correto para o índice do FAISS no arquivo `.env`.

    ```python
    db = FAISSDatabase()
    ```

    O output deve ser algo como:
    ```
    FAISS index loaded successfully
    Index has X embeddings
    Documents loaded successfully
    Loaded X documents
    ```

4. Realize uma busca:

    ```python
    query = "O que é o Transtorno de Déficit de Atenção/Hiperatividade (TDAH)"
    retorno = db.search(query)
    print(retorno)
    ```

O output deve ser algo como:

    ```json
[
    (
        {
            'id': None,
            'metadata': {'source': 'DSM5_organized.pdf', 'start_index': 2830},
            'page_content': 'Transtorno de déficit de atenção/hiperatividade. O transtorno específico da aprendizagem \ndistingue-se do desempenho acadêmico insatisfatório associado ao TDAH, porque nessa condi-\nção os problemas podem não necessariamente refletir dificuldades específicas na aprendizagem de habilidades, podendo, sim, ser reflexo de dificuldades no desempenho daquelas habilidades. Todavia, a comorbidade de transtorno específico da aprendizagem e TDAH é mais frequente do que o esperado apenas.',
            'type': 'Document'
        },
        0.2293765
    ),
    ...
]
    ```

## Metodologia

### Tratamento de Dados

#### DSM-5

O DSM-5 é um manual de classificação de transtornos mentais que é amplamente utilizado por profissionais de saúde mental. O documento possui quase 1000 páginas. O tratamento e a extração das informações do DSM-5 estão no módulo `DSM5`, mais precisamente no arquivo [process_dsm5_pdf.py](mental_health_ai/db_knowledge/DSM5/process_dsm5_pdf.py).

##### Abordagem

O processo de extração e preparação das informações do DSM-5 é realizado utilizando o seguinte método:

1. **Carregamento do PDF**:
   Utilizamos a biblioteca `PyPDFLoader` para carregar o documento PDF do DSM-5. O arquivo PDF é especificado pelo caminho definido em `FULL_PATH`.

    ```python
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(FULL_PATH)
    documents = loader.load()
    ```

2. **Divisão em Sentenças**:
   O conteúdo de cada página do PDF é dividido em sentenças usando o `nltk`, uma biblioteca de processamento de linguagem natural. Isso nos ajuda a garantir que as informações não sejam cortadas no meio de frases importantes.

    ```python
    import nltk
    from .utils import split_into_sentences

    nltk.download('punkt')
    sentence_splits = [split_into_sentences(doc) for doc in documents]
    sentence_splits = [sentence for sublist in sentence_splits for sentence in sublist]
    ```

3. **Reconstrução dos Documentos**:
   As sentenças são então reconstruídas em documentos menores com um número específico de linhas por chunk. Este passo é essencial para criar trechos de texto que podem ser facilmente indexados e buscados.

    ```python
    from .utils import reconstruct_documents

    TARGET_LINES_PER_CHUNK = 5
    reconstructed_docs = reconstruct_documents(sentence_splits, target_lines_per_chunk=TARGET_LINES_PER_CHUNK)
    ```

4. **Criação dos Documentos**:
   Os chunks reconstruídos são convertidos em objetos `Document` do LangChain, contendo o conteúdo da página e metadados adicionais, como a origem e o índice de início.

    ```python
    from langchain.schema import Document

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
    ```

5. **Salvamento dos Documentos**:
   Finalmente, os documentos são salvos em um arquivo JSON para uso posterior.

    ```python
    import json

    splitted_docs_json = [doc.dict() for doc in splitted_documents]

    with open(f'{DATA_PATH}/DSM5_splitted_documents.json', 'w', encoding='utf-8') as f:
        json.dump(splitted_docs_json, f, ensure_ascii=False)

    print("Splitted documents saved to 'splitted_documents.json'")
    ```
