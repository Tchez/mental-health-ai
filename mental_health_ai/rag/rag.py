from rich import print

from mental_health_ai.rag.database.db_interface import DatabaseInterface
from mental_health_ai.rag.llm.llm_interface import LLMInterface


class RAGFactory:
    """
    Factory class for the Retrieval-Augmented Generation (RAG) model.

    This class combines a vector database retriever and a language model to generate responses to user queries.

    Attributes:
        vector_db (DatabaseInterface): The vector database instance used for retrieving relevant documents.
        llm (LLMInterface): The language model instance used for generating responses.

    Examples:
        >>> from mental_health_ai.rag.database.weaviate_impl import WeaviateClient
        >>> from mental_health_ai.rag.llm.openai_impl import OpenAILLM
        >>> vector_db = WeaviateClient()
        >>> llm = OpenAILLM()
        >>> rag_factory = RAGFactory(vector_db=vector_db, llm=llm)
        >>> query = 'Responda em um parágrafo, o que é o TDAH?'
        >>> response, documents = rag_factory.generate_response(query)
        >>> print(f'Response: {response}')
    """  # noqa: E501

    def __init__(self, vector_db: DatabaseInterface, llm: LLMInterface):
        self.vector_db = vector_db
        self.llm = llm

    @staticmethod
    def _get_documents_by_contexts(documents: list) -> tuple[list, list]:
        """
        Categorize documents by their types (DSM-5 and articles).

        Args:
            documents (list): List of retrieved documents.

        Returns:
            tuple[list, list]: A tuple containing two lists:
                - dsm5_docs: Documents of type 'DSM-5'.
                - articles: Documents of type 'article'.
        """
        dsm5_docs = [
            doc
            for doc in documents
            if doc.properties['metadata'].get('type', '').lower() == 'dsm-5'
        ]

        articles = [
            doc
            for doc in documents
            if doc.properties['metadata'].get('type', '').lower() == 'article'
        ]

        return dsm5_docs, articles

    @staticmethod
    def _format_document_context_single(doc, content: str) -> str:
        """
        Format a single document's content and metadata into a string.

        Args:
            doc: A document object.
            content (str): The concatenated content for the page.

        Returns:
            str: Formatted string containing the document's content and metadata.
        """  # noqa: E501
        title = doc.properties.get('title', 'No Title')
        metadata = doc.properties.get('metadata', {})
        page_number = metadata.get('page_number', 'N/A')
        source = metadata.get('source', 'N/A')
        source_description = metadata.get('source_description', 'N/A')
        metadata = doc.properties.get('metadata', {})

        formatted_context = (
            f'Título: {title}\n'
            f'Número da página: {page_number}\n'
            f'Fonte: {source}\n'
            f'Descrição da fonte: {source_description}\n'
            f'Metadados: {metadata}\n'
            f'Content:\n{content}\n'
            '---------------------\n'
        )

        return formatted_context

    def _gather_dsm5_context(self, dsm5_docs: list) -> str:
        """
        Gather context from DSM-5 documents.

        Args:
            dsm5_docs (list): List of DSM-5 documents.

        Returns:
            str: Combined context from DSM-5 documents.

        Raises:
            Exception: If no DSM-5 context is found.
        """
        if not dsm5_docs:
            print('[red]Nenhum documento DSM-5 encontrado![/red]')
            raise Exception('No DSM-5 documents found.')

        page_numbers = {
            doc.properties['metadata'].get('page_number')
            for doc in dsm5_docs
            if doc.properties['metadata'].get('page_number') is not None
        }

        full_context = []

        for page_number in page_numbers:
            all_docs_for_page = (
                self.vector_db.get_documents_by_type_and_page_number(
                    doc_type='dsm-5', page_number=page_number
                )
            )

            if not all_docs_for_page:
                print(
                    f'[yellow]No documents found for page {page_number}.[/yellow]'  # noqa: E501
                )
                continue

            concatenated_content = '\n'.join(
                doc.properties.get('page_content', '')
                for doc in all_docs_for_page
            )

            first_doc = all_docs_for_page[0]
            formatted_context = self._format_document_context_single(
                doc=first_doc, content=concatenated_content
            )
            full_context.append(formatted_context)

        if not full_context:
            print('[red]Nenhum contexto DSM-5 encontrado![/red]')
            raise Exception('No DSM-5 context found.')

        return '\n'.join(full_context)

    def _gather_article_context(self, article_docs: list) -> str:
        """
        Gather context from articles.

        Args:
            article_docs (list): List of article documents.

        Returns:
            str: Combined context from article documents.
        """
        if not article_docs:
            print('[red]Nenhum documento de artigo encontrado![/red]')
            raise Exception('No article documents found.')

        article_and_page = {
            (
                doc.properties['metadata'].get('source'),
                doc.properties['metadata'].get('page_number'),
            )
            for doc in article_docs
        }

        full_context = []

        for source, page_number in article_and_page:
            all_docs_for_page = (
                self.vector_db.get_documents_by_type_and_page_number(
                    doc_type='article', page_number=page_number, source=source
                )
            )

            if not all_docs_for_page:
                print(
                    f'[yellow]No documents found for page {page_number} of {source}.[/yellow]'  # noqa: E501
                )
                continue

            concatenated_content = '\n'.join(
                doc.properties.get('page_content', '')
                for doc in all_docs_for_page
            )

            first_doc = all_docs_for_page[0]
            formatted_context = self._format_document_context_single(
                doc=first_doc, content=concatenated_content
            )
            full_context.append(formatted_context)

        if not full_context:
            print('[red]Nenhum contexto de artigos encontrado![/red]')
            raise Exception('No article context found.')

        return '\n'.join(full_context)

    def _handle_contexts(self, documents: list) -> str:
        """
        Handle contexts for all types of documents.

        Args:
            documents (list): List of retrieved documents.

        Returns:
            str: Combined context from all document types.

        Raises:
            Exception: If no context is found.
        """
        dsm5_docs, article_docs = self._get_documents_by_contexts(documents)
        context_parts = []

        if dsm5_docs:
            dsm5_context = self._gather_dsm5_context(dsm5_docs)
            if dsm5_context:
                context_parts.append(dsm5_context)

        if article_docs:
            article_context = self._gather_article_context(article_docs)
            if article_context:
                context_parts.append(article_context)

        if not context_parts:
            print('[red]Nenhum contexto encontrado![/red]')
            raise Exception('No context found.')  # noqa: E501

        return '\n\n'.join(context_parts)

    def generate_response(
        self, query: str, top_k: int = 5
    ) -> tuple[str, list]:
        """
        Generates a response to a given query using the RAG model.

        Args:
            query (str): The query to generate a response for.
            top_k (int, optional): The number of documents to retrieve from the database. Defaults to 5.

        Returns:
            tuple[str, list]: The response generated by the RAG model and the list of retrieved documents.

        Raises:
            Exception: If the database is not available or no documents are found.
        """  # noqa: E501
        if not self.vector_db.verify_database():
            print('[red]O banco de dados não está disponível![/red]')
            raise Exception('Database not available or empty.')

        retrieved_documents = self.vector_db.search(query, limit=top_k)

        if not retrieved_documents:
            print('[red]Nenhum documento encontrado![/red]')
            raise Exception('No documents found.')  # noqa: E501

        context = self._handle_contexts(retrieved_documents)
        print(f'Context: {context}')

        system_context = f"""Papel: Você é um chatbot especializado em saúde mental que receberá um contexto com informações confiáveis relacionadas à pergunta do usuário, provenientes de uma base de dados vetorial.
Regras:
    - Você não é um profissional de saúde e não pode fornecer diagnósticos ou tratamentos;
    - O conteúdo fornecido pode estar segmentado e fora de ordem; ao responder, organize as informações de forma coerente e cite a fonte de forma humanizada e fácil de entender (ex.: não apenas o nome do pdf, mas sim o nome do artigo/livro/...);
    - Você pode utilizar o contexto para fornecer informações embasadas e verdadeiras. Caso o contexto não seja suficiente, você deve informar ao usuário, mas nunca inventar informações;
    - Detalhe bem suas respostas, mas mantenha-as certas, não invente informações.
    - Ao final de todas as respostas, mencione as fontes utilizadas para a resposta. No caso de artigos, mencione o nome do artigo e outras informações relevantes para que o usuário possa acessar a fonte original.
    - Apenas referencie na resposta os contextos passados dentro da tag <contexto>. E caso o contexto seja de um artigo e o texto cite uma referência, não cite-a como se tivesse acesso à ela pois você só conhece o texto passado na tag contexto.
    - Ao citar as fontes no final da pergunta, apenas cite as que realmente foram úteis para o texto.

<contexto>{context}</contexto>"""  # noqa: E501

        messages = [
            ('system', system_context),
            ('human', f'Pergunta: {query}\n\nResposta:'),
        ]

        response = self.llm.generate_response(messages)
        return response, retrieved_documents


if __name__ == '__main__':
    from mental_health_ai.rag.database.weaviate_impl import WeaviateClient
    from mental_health_ai.rag.llm.openai_impl import OpenAILLM

    with WeaviateClient() as vector_db:
        llm = OpenAILLM()
        rag_factory = RAGFactory(vector_db=vector_db, llm=llm)
        query = 'Responda em um parágrafo, o que é o TDAH?'  # noqa: E501
        response, documents = rag_factory.generate_response(query)
        print(f'Response: {response}')
