# Projeto Mental Health AI

> Projeto em desenvolvimento...

Esse projeto é meu trabalho de conclusão do curso de Ciência da Computação. O objetivo é desenvolver e validar o uso da arquitetura Retrieval-Augmented Generation (RAG) para criar um chatbot informativo sobre saúde mental.

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

## Metodologia Utilizada

### Extração e Tratamento de Dados

#### DSM-5

O DSM-5 é um manual de classificação de transtornos mentais amplamente utilizado por profissionais de saúde mental. O tratamento e a extração das informações do DSM-5 estão detalhados no módulo `DSM5`, especificamente no arquivo [process_dsm5_pdf.py](mental_helth_ai/processing_raw_data/process_dsm5_pdf.py).

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
    def split_into_sentences(doc):
        """
        Splits the document content into sentences using NLTK's sentence tokenizer.

        Args:
        doc (Document): The document to be split.

        Returns:
        List[str]: List of sentences.
        """
        return nltk.sent_tokenize(doc.page_content)
    ```

3. **Reconstrução dos Documentos**:
   As sentenças são reconstruídas em documentos menores com um número específico de linhas por chunk. Este passo cria trechos de texto para serem indexados e buscados.

    ```python
    def reconstruct_documents(sentences, target_lines_per_chunk=15):
        """
        Reconstructs documents from sentences, ensuring that chunks have
        approximately the target number of lines.

        Args:
        sentences (List[str]): List of sentences to be grouped into chunks.
        target_lines_per_chunk (int): Target number of lines per chunk.

        Returns:
        List[str]: List of document chunks.
        """
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
   Os chunks reconstruídos são convertidos em objetos que seguem o padrão do `schema` do projeto, contendo informações como o conteúdo da página, o número da página e a fonte do documento.

    ```python
    for idx, chunk in enumerate(reconstructed_docs):
        splitted_documents.append({
            'title': f'DSM-5 Page {page_number}',
            'page_content': chunk,
            'metadata': {
                'type': 'DSM-5',
                'source': FILE_NAME,
                'page_number': page_number,
                'source_description': 'O Manual Diagnóstico e Estatístico de Transtornos Mentais 5.ª edição, ou DSM-5, é um manual diagnóstico e estatístico feito pela Associação Americana de Psiquiatria para definir como é feito o diagnóstico de transtornos mentais. Usado por psicólogos, fonoaudiólogos, médicos e terapeutas ocupacionais. A versão atualizada saiu em maio de 2013 e substitui o DSM-IV criado em 1994 e revisado em 2000. Desde o DSM-I, criado em 1952, esse manual tem sido uma das bases de diagnósticos de saúde mental mais usados no mundo.',
                'date': '2013-05-18T00:00:00Z',
            },
        })
    ```

    

5. **Salvamento dos Documentos**:
   Finalmente, os documentos são salvos em um arquivo JSON para serem carregados e indexados posteriormente.

    ```python
    splitted_documents = [doc for doc in splitted_documents if doc['page_content'].strip()]

    with open(f'{OUTPUT_PATH}dsm5.json', 'w', encoding='utf-8') as f:
        json.dump(splitted_documents, f, ensure_ascii=False)
    ```

6. **Indexação dos Documentos**:

   Com os jsons dos documentos prontos, é possível indexá-los no banco vetorial. Para isso, é necessário instanciar alguma implementação da interface [DatabaseInterface](mental_helth_ai/rag/database/db_interface.py) e chamar o método `add_document` passando o documento ou carregando documentos de um diretório de arquivos JSONs utilizando o método `load_documents`.

    ```python
    db = DatabaseImplementation()
    db.load_documents('data/processed/')
    ```


#### CID-10

O CID-10 é a décima revisão da Classificação Estatística Internacional de Doenças e Problemas Relacionados à Saúde. 

##### Abordagem

O processo de extração e preparação das informações do CID-10 segue os passos abaixo:

1. **Download e Filtragem da Tabela**:
   A planilha do CID-10 foi baixada e filtrada pelos códigos F00 a F99, que correspondem aos transtornos mentais e comportamentais.

2. **Extração de Informações**:
   As informações relevantes foram extraídas da planilha, incluindo o código, o nome do transtorno e a descrição.

> **Nota**: O tratamento dos dados do CID-10 ainda está em andamento...

#### Artigos e Fontes Externas

Além dos documentos do DSM-5 e CID-10, também fiz a coleta de artigos e informações de fontes externas para enriquecer a base de conhecimento do chatbot. A abordagem para esses documentos é semelhante à do DSM-5, com a diferença de que os artigos são extraídos utilizando web scraping e para os metadados, utilizei um json com os meta dados de cada artigo. O arquivo responsável pelo scraping é o [article_scraper.py](mental_helth_ai/processing_raw_data/article_scraper.py) e o arquivo responsável por processar os artigos e seus metadados é o [process_articles.py](mental_helth_ai/processing_raw_data/process_articles.py).

> **Nota**: O tratamento dos dados dos artigos e fontes externas ainda está em andamento...

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

    Crie um arquivo `.env` na raiz do projeto com base no arquivo `.env.example` e altere os valores conforme necessário:

    > Lembre-se que as variáveis de ambiente mudam conforme o banco de dados, modelo de embedding e LLM que você escolher utilizar.

    > No momento, o projeto está configurado para utilizar o Weaviate como banco de dados e o modelo de embedding da OpenAI ou da lib `sentence-transformers`.

### Uso

1. Ative o ambiente virtual:
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

4. Faça o carregamento dos documentos já processados:
    > Tenha em mente que esse comando irá carregar todos os documentos processados no diretório `data/processed/`. Isso pode levar um bom tempo, e caso esteja utilizando um modelo de embedding pago -- como o da OpenAI -- pode consumir muitos créditos.

    ```python
    db.load_documents('data/processed/')
    ```

5. Realize uma busca:

    ```python
    query = "O que é o Transtorno de Déficit de Atenção/Hiperatividade (TDAH)"
    retorno = db.search(query)
    print(retorno)
    ```

    O output deve ser algo como:

    ```python
    [
        Object(
            uuid=_WeaviateUUIDInt('60ac1a87-4b0a-4a62-b4ab-ed56cf5d363c'),
            metadata=MetadataReturn(creation_time=None, last_update_time=None, distance=0.09966814517974854, certainty=None, score=0.0, explain_score=None, is_consistent=None, rerank_score=None),
            properties={
                'title': 'DSM-5 Page 93',
                'metadata': {
                    'source_description': 'O Manual Diagnóstico e Estatístico de Transtornos Mentais 5.ª edição, ou DSM-5, é um manual diagnóstico e estatístico feito pela Associação Americana de Psiquiatria para definir como é feito o diagnóstico de transtornos mentais. Usado por psicólogos, fonoaudiólogos, médicos e terapeutas ocupacionais. A versão atualizada saiu em maio de 2013 e substitui o DSM-IV criado em 1994 e revisado em 2000. Desde o DSM-I, criado em 1952, esse manual tem sido uma das bases de diagnósticos de saúde mental mais usados no mundo.',
                    'date': datetime.datetime(2013, 5, 18, 0, 0, tzinfo=datetime.timezone.utc),
                    'type': 'DSM-5',
                    'source': 'DSM5_organized.pdf',
                    'page_number': 93.0
                },
                'page_content': 'Transtorno de Déficit de Atenção/Hiperatividade 61\nCaracterísticas Diagnósticas\nA característica essencial do transtorno de déficit de atenção/hiperatividade é um padrão persis-\ntente de desatenção e/ou hiperatividade-impulsividade que interfere no funcionamento ou no desenvolvimento. A desatenção manifesta-se comportamentalmente no TDAH como divagação em tarefas, falta de persistência, dificuldade de manter o foco e desorganização – e não constitui consequência de desafio ou falta de compreensão. A  hiperatividade  refere-se a atividade motora \nexcessiva (como uma criança que corre por tudo) quando não apropriado ou remexer, batucar ou conversar em excesso. Nos adultos, a hiperatividade pode se manifestar como inquietude extre-ma ou esgotamento dos outros com sua atividade. A impulsividade  refere-se a ações precipitadas \nque ocorrem no momento sem premeditação e com elevado potencial para dano à pessoa (p. ex., atravessar uma rua sem olhar). A impulsividade pode ser reflexo de um desejo de recompensas imediatas ou de incapacidade de postergar a gratificação. Comportamentos impulsivos podem se manifestar com intromissão social (p. ex., interromper os outros em excesso) e/ou tomada de decisões importantes sem considerações acerca das consequências no longo prazo (p. ex., assu-mir um emprego sem informações adequadas).'
            },
            references=None,
            vector={},
            collection='Documents'
        ),
        ...
    ]
    ```

## Modelagem do Banco de Dados

### Weaviate

Para organizar e buscar informações dentro do Weaviate, foi criada uma collection chamada `Documents`. Essa collection armazena diferentes tipos de documentos, como artigos, informações do DSM-5, e dados do CID-10, utilizando um modelo de dados estruturado para otimizar a busca e a classificação dos documentos. Abaixo estão os detalhes dos campos utilizados na coleção `Documents`:

- title (TEXT): Este campo armazena o título do documento, como o título de um artigo ou o nome de uma condição do DSM-5 ou CID-10. Serve para identificar rapidamente o documento durante a busca.

- page_content (TEXT): Contém o conteúdo principal do documento. Esse campo é o mais importante para operações de busca e similaridade, pois armazena o texto que será vetorizado e utilizado nas consultas.

- metadata (OBJECT): Campo que agrupa metadados adicionais do documento. Dentro deste objeto, há várias propriedades aninhadas que ajudam a contextualizar e classificar melhor os documentos:

- type (TEXT): Indica o tipo do documento (e.g., artigo, DSM-5, CID-10). Isso é útil para filtrar documentos por categoria durante a busca.

- source (TEXT): Armazena a origem do documento, como o URL ou nome do site. Esse campo é essencial para rastreabilidade, especialmente para documentos online.

- page_number (NUMBER): Referencia o número da página de onde o conteúdo foi extraído, útil para documentos que possuem múltiplas páginas, como o DSM-5 ou artigos em PDF.

- source_description (TEXT): Descreve brevemente o contexto ou importância da fonte, como uma breve descrição de um artigo, uma entrada do DSM-5 ou CID-10.

- date (DATE): Armazena a data de publicação do documento, útil para ordenar e filtrar documentos por tempo. **No formato RFC3339.**


## Próximos Passos

- [ ] Finalizar o tratamento dos dados do CID-10 e dos artigos.
    - [ ] Criar scripts para extrair e tratar os dados do CID-10.
    - [ ] Implementar um web scraper para coletar artigos de fontes externas, salvando os metadados em arquivos .txt.
    - [ ] Processar os artigos e os metadados em arquivos JSON para serem indexados no banco de dados.
- [ ] Implementar uma API usando FastAPI para disponibilizar as funcionalidades do chatbot.
- [ ] Dockerizar o projeto para facilitar a execução e o deploy.
- [ ] Testar e validar a precisão das respostas do chatbot.