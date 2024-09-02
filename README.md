# Mental Health AI

> **Desenvolvendo um Chatbot Informativo para Saúde Mental usando Retrieval-Augmented Generation (RAG)**

Este projeto é o Trabalho de Conclusão de Curso (TCC) em Ciência da Computação. O objetivo principal é desenvolver e validar uma arquitetura de **Retrieval-Augmented Generation (RAG)** para criar um chatbot que forneça informações precisas e úteis sobre saúde mental.

## Sumário

- [Metodologia Utilizada](#metodologia-utilizada)
  - [Extração e Tratamento de Dados](#extração-e-tratamento-de-dados)
    - [DSM-5](#dsm-5)
    - [CID-10](#cid-10)
    - [Artigos e Fontes Externas](#artigos-e-fontes-externas)
- [Utilização do Projeto](#utilização-do-projeto)
  - [Requisitos](#requisitos)
  - [Instalação](#instalação)
  - [Uso](#uso)
- [Modelagem do Banco de Dados](#modelagem-do-banco-de-dados)
- [Próximos Passos](#próximos-passos)
- [Contribuições](#contribuições)
- [Licença](#licença)

## Metodologia Utilizada

### Extração e Tratamento de Dados

#### DSM-5

O DSM-5 é um manual de classificação de transtornos mentais amplamente utilizado por profissionais de saúde mental. O tratamento e a extração das informações do DSM-5 são realizados no módulo `DSM5`, especificamente no arquivo [process_dsm5_pdf.py](mental_helth_ai/processing_raw_data/process_dsm5_pdf.py).

##### Abordagem

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
    def split_into_sentences(doc):
        return nltk.sent_tokenize(doc.page_content)
    ```

3. **Reconstrução dos Documentos**:
   As sentenças são reorganizadas em documentos menores, com um número específico de linhas por chunk. Esse processo cria trechos de texto otimizados para indexação e busca.

    ```python
    def reconstruct_documents(sentences, target_lines_per_chunk=15):
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split('\n'))
            if current_length + sentence_length > target_lines_per_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks
    ```

4. **Criação dos Documentos**:
   Os chunks são convertidos em objetos que seguem o padrão do `schema` do projeto, contendo informações detalhadas, como o conteúdo da página, o número da página e a fonte do documento.

5. **Salvamento e Indexação**:
   Os documentos são salvos em um arquivo JSON e posteriormente indexados no banco de dados vetorial, utilizando uma implementação da interface [DatabaseInterface](mental_helth_ai/rag/database/db_interface.py).

#### CID-10

O CID-10 é a décima revisão da Classificação Estatística Internacional de Doenças e Problemas Relacionados à Saúde. 

##### Abordagem

1. **Download e Filtragem da Tabela**:
   A planilha do CID-10 foi baixada e filtrada pelos códigos F00 a F99, que correspondem aos transtornos mentais e comportamentais.

2. **Extração de Informações**:
   As informações relevantes foram extraídas da planilha, incluindo o código, o nome do transtorno e a descrição.

> **Nota**: O tratamento dos dados do CID-10 ainda está em andamento.

#### Artigos e Fontes Externas

Além dos documentos do DSM-5 e CID-10, também foi realizada a coleta de artigos e informações de fontes externas para enriquecer a base de conhecimento do chatbot.

##### Abordagem

1. **Web Scraping**:
   Utilizei um scraper personalizado ([article_scraper.py](mental_helth_ai/processing_raw_data/article_scraper.py)) para extrair artigos da base de dados SciELO. O scraper percorreu **388** páginas e conseguiu extrair metadados de **5822** artigos no total. Os metadados extraídos incluem título, link do artigo, link para o PDF, uma breve descrição e a data de publicação. Estes dados foram armazenados em um arquivo JSON (`articles_metadata.json`).

2. **Tratamento dos Dados**:
   Durante o processo de extração, implementei uma lógica para validar as datas, garantindo que apenas anos válidos fossem capturados. Além disso, o scraper foi configurado para salvar automaticamente os dados extraídos em caso de interrupção, evitando a perda de progresso.

3. **Backup e Processamento Seletivo**:
   Como o volume de artigos é significativo, um backup dos metadados foi salvo em `articles_metadata.json`. Para o próximo passo, planejo processar apenas os 100 primeiros artigos desse arquivo, devido às limitações de tempo e recursos necessários para o processamento completo de todos os artigos.

   Utilizei o arquivo `articles_metadata.json` para baixar os PDFs dos 100 primeiros artigos. O download dos PDFs foi realizado com sucesso, e os arquivos foram salvos localmente para processamento posterior.

4. **Processamento dos Artigos**:
    Tendo os pdfs e o json de metadados, o próximo passo é processar os artigos para extrair o conteúdo e salvar em um formato adequado para indexação no banco de dados, seguindo a mesma abordagem utilizada para o DSM-5.

> **Nota**: O tratamento dos artigos ainda está em andamento.

## Utilização do Projeto

### Requisitos

- [Python 3.12.*](https://www.python.org/downloads/release/python-3120/)
- [Poetry](https://python-poetry.org/docs/)
- [Docker](https://docs.docker.com/get-started/get-docker/)

### Instalação

1. Clone o repositório:
    ```sh
    git clone https://github.com/Tchez/mental-health-ai.git
    cd mental-health-ai
    ```

2. Configure as variáveis de ambiente:

    Crie um arquivo `.env` na raiz do projeto com base no arquivo `.env.example` e altere os valores conforme necessário.

    > As variáveis de ambiente podem variar conforme o banco de dados, modelo de embedding e LLM que você escolher utilizar.

3. Suba o banco de dados Weaviate usando Docker:
    ```sh
    docker-compose up -d
    ```
> **Nota**: Caso queira usar um modelo de embedding local, é necessário configurar nas variáveis de ambiente e utilizar o seguinte comando: `docker-compose -f 'docker-compose-local-embedding.yml' up -d`.

4. Instale as dependências do projeto:
    ```sh
    poetry install
    ```

### Uso

1. Garanta que o ambiente virtual do Poetry esteja ativado:
    ```sh
    poetry shell
    ```

2. Rode o arquivo da implementação do banco de dados de maneira interativa:
    ```sh
    python -i mental_helth_ai/rag/database/weaviate_impl.py
    ```

3. Crie uma instância do banco de dados:
    > Lembre-se de colocar os valores corretos das variáveis de ambiente no arquivo `.env`.

    ```python
    db = WeaviateClient()
    ```

4. Carregue os documentos processados:
    ```python
    db.load_documents('data/processed/')
    ```

5. Realize uma busca:
    ```python
    query = "O que é o Transtorno de Déficit de Atenção/Hiperatividade (TDAH)"
    retorno = db.search(query)
    print(retorno)
    ```

### Estrutura de Diretórios

```bash
mental-health-ai/
│
├── data/
│   ├── raw/          # Dados brutos, como PDFs e arquivos JSON dos artigos extraídos
│   └── processed/    # Dados processados e prontos para indexação
│
├── mental_helth_ai/
│   ├── processing_raw_data/  # Scripts para extração e processamento de dados
│   └── rag/                  # Implementação da arquitetura RAG (base de dados, modelos, etc.)
│
├── tests/          # Testes unitários e de integração
│
└── README.md         # Documentação do projeto
...
```

## Modelagem do Banco de Dados

### Weaviate

Para organizar e buscar informações dentro do Weaviate, foi criada uma collection chamada `Documents`, que armazena diferentes tipos de documentos, como artigos, informações do DSM-5 e dados do CID-10, utilizando um modelo de dados estruturado para otimizar a busca e a classificação dos documentos.

- **title (TEXT)**: Título do documento.
- **page_content (TEXT)**: Conteúdo principal do documento, utilizado para operações de busca e similaridade.
- **metadata (OBJECT)**: Campo que agrupa metadados adicionais do documento.
  - **type (TEXT)**: Tipo do documento (e.g., artigo, DSM-5, CID-10).
  - **source (TEXT)**: Origem do documento (e.g., URL ou nome do site).
  - **page_number (NUMBER)**: Número da página de onde o conteúdo foi extraído.
  - **source_description (TEXT)**: Descrição do contexto ou importância da fonte.
  - **date (DATE)**: Data de publicação do documento no formato RFC3339.

## Próximos Passos

- [ ] Finalizar o tratamento dos dados do CID-10 e dos artigos.
    - [ ] Processar os artigos e os metadados em arquivos JSON para serem indexados no banco de dados.
- [ ] Implementar uma API usando FastAPI para disponibilizar as funcionalidades do chatbot.
- [ ] Dockerizar o projeto para facilitar a execução e o deploy.
- [ ] Testar e validar a precisão das respostas do chatbot.

## Contribuições

Contribuições para este projeto são bem-vindas! Se você encontrar algum problema ou tiver sugestões de melhoria, sinta-se à vontade para abrir uma issue ou enviar um pull request.

## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).
