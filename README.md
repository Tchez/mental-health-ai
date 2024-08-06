# Projeto Mental Health AI

> Projeto em desenvolvimento

Esse projeto é meu trabalho de conclusão do curso de Ciência da Computação. O objetivo é desenvolver e validar o uso da arquitetura Retrieval-Augmented Generation (RAG) para criar um chatbot informativo sobre saúde mental.

## Sumário

- [Metodologia Utilizada](#metodologia-utilizada)
  - [Tratamento de Dados](#tratamento-de-dados)
    - [DSM-5](#dsm-5)
- [Utilização do Projeto](#utilização-do-projeto)
  - [Requisitos](#requisitos)
  - [Instalação](#instalação)
  - [Uso](#uso)
- [Próximos Passos](#próximos-passos)

## Metodologia Utilizada

### Tratamento de Dados

#### DSM-5

O DSM-5 é um manual de classificação de transtornos mentais amplamente utilizado por profissionais de saúde mental. O tratamento e a extração das informações do DSM-5 estão detalhados no módulo `DSM5`, especificamente no arquivo [process_dsm5_pdf.py](mental_helth_ai/db_knowledge/DSM5/process_dsm5_pdf.py).

##### Abordagem

O processo de extração e preparação das informações do DSM-5 segue os passos abaixo:

1. **Carregamento do PDF**:
   Utilizei a biblioteca `PyPDFLoader` do LangChain para carregar o documento PDF do DSM-5.

    ```python
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(FULL_PATH)
    documents = loader.load()
    ```

2. **Divisão em Sentenças**:
   O conteúdo de cada página do PDF é dividido em sentenças usando a biblioteca `nltk`, garantindo que as informações não sejam cortadas no meio de frases importantes.

    ```python
    import nltk
    from .utils import split_into_sentences

    nltk.download('punkt')
    sentence_splits = [split_into_sentences(doc) for doc in documents]
    sentence_splits = [sentence for sublist in sentence_splits for sentence in sublist]
    ```

3. **Reconstrução dos Documentos**:
   As sentenças são reconstruídas em documentos menores com um número específico de linhas por chunk. Este passo cria trechos de texto para serem indexados e buscados.

    ```python
    from .utils import reconstruct_documents

    TARGET_LINES_PER_CHUNK = 15
    reconstructed_docs = reconstruct_documents(
        sentence_splits, target_lines_per_chunk=TARGET_LINES_PER_CHUNK
    )
    ```

4. **Criação dos Documentos**:
   Os chunks reconstruídos são convertidos em objetos `Document` do LangChain, contendo o conteúdo da página e metadados adicionais, incluindo o número da página.

    ```python
    from langchain.schema import Document

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
                        'source': FILE_NAME,
                        'start_index': idx * TARGET_LINES_PER_CHUNK,
                        'page_number': page_number
                    },
                )
            )

    splitted_documents = [doc for doc in splitted_documents if doc.page_content.strip()]
    ```

5. **Salvamento dos Documentos**:
   Finalmente, os documentos são salvos em um arquivo JSON para uso futuro.

    ```python
    import json

    splitted_docs_json = [doc.dict() for doc in splitted_documents]

    with open(f'{DATA_PATH}/DSM5_splitted_documents.json', 'w', encoding='utf-8') as f:
        json.dump(splitted_docs_json, f, ensure_ascii=False)

    print("Splitted documents saved to 'splitted_documents.json'")
    ```

6. **Indexação dos Documentos**:

   Os documentos salvos em formato JSON são carregados e indexados no banco de dados FAISS para permitir a busca rápida de informações. O arquivo [loading_data_from_json.py](mental_helth_ai/db_knowledge/DSM5/loading_data_from_json.py) contém o código que faz esse carregamento para os documentos do DSM-5, salvando os arquivos em lotes (de 500 em 500), adicionando-os no json de documentos e gerando um banco de dados de índice FAISS, utilizando os caminhos especificados como variáveis de ambiente.


#### CID-10

O CID-10 é a décima revisão da Classificação Estatística Internacional de Doenças e Problemas Relacionados à Saúde. 

##### Abordagem

O processo de extração e preparação das informações do CID-10 segue os passos abaixo:

1. **Download e Filtragem da Tabela**:
   A planilha do CID-10 foi baixada e filtrada pelos códigos F00 a F99, que correspondem aos transtornos mentais e comportamentais.

2. **Extração de Informações**:
   As informações relevantes foram extraídas da planilha, incluindo o código, o nome do transtorno e a descrição.

> **Nota**: O tratamento dos dados do CID-10 ainda está em andamento...

## Utilização do Projeto

### Requisitos

- Python 3.12.*
- Poetry

### Instalação

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
        {
            "id": None,
            "metadata": {
                "source": "DSM5_organized.pdf",
                "start_index": 2830,
                "page_number": 189
            },
            "page_content": "Transtorno de déficit de atenção/hiperatividade. O transtorno específico da aprendizagem \ndistingue-se do desempenho acadêmico insatisfatório associado ao TDAH, porque nessa condi-\nção os problemas podem não necessariamente refletir dificuldades específicas na aprendizagem de habilidades, podendo, sim, ser reflexo de dificuldades no desempenho daquelas habilidades. Todavia, a comorbidade de transtorno específico da aprendizagem e TDAH é mais frequente do que o esperado apenas.",
            "type": "Document"
        },
        0.2293765
    ]
    ```

## Próximos Passos

- [ ] Criar Interface de LLM
- [ ] Criar Implementação de LLM local
- [ ] Criar Implementação de LLM utilizando a OpenAI API
- [ ] Criar Fábrica RAG que recebe implementações das interfaces de LLM e Retriever para retornar uma classe RAG
- [ ] Criar lógica de pergunta para classe RAG
    - [ ] Chamar o Retriever para buscar documentos relacionados
    - [ ] Montar a pergunta para o LLM passando os documentos relacionados
    - [ ] Chamar o LLM para responder a pergunta
- [ ] Implementar uma API usando FastAPI para disponibilizar as funcionalidades do chatbot.
- [ ] Testar e validar a precisão das respostas do chatbot.

Estou desenvolvendo o projeto com base na metodologia e nas ferramentas descritas, e continuo a adicionar detalhes e atualizações à medida que avanço no desenvolvimento.