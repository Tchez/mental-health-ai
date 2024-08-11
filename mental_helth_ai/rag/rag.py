from rich import print

from mental_helth_ai.rag.database.db_interface import DatabaseInterface
from mental_helth_ai.rag.llm.llm_interface import LLMInterface


class RAGFactory:
    """Factory class for the RAG model.

    Attributes:
        retriever (DatabaseInterface): The retriever instance to use.
        llm (LLMInterface): The language model instance to use.

    Examples:
        >>> from mental_helth_ai.rag.database.faiss_db_impl import FAISSDatabase
        >>> from mental_helth_ai.rag.llm.ollama_impl import OllamaLLM
        >>> retriever = FAISSDatabase()
        >>> llm = OllamaLLM()
        >>> rag_factory = RAGFactory(retriever=retriever, llm=llm)
        >>> query = 'Responda em um parágrafo, o que é o Transtorno de Déficit de Atenção/Hiperatividade (TDAH)?'
        >>> response = rag_factory.generate_response(query)
        >>> print(f'Response: {response}')
    """  # noqa: E501

    def __init__(self, vector_db: DatabaseInterface, llm: LLMInterface):
        self.vector_db = vector_db
        self.llm = llm

    def generate_response(self, query: str, top_k: int = 5) -> str:
        retrieved_documents = self.vector_db.search(query, limit=top_k)

        print('Retrieved documents:')
        print(retrieved_documents)

        context = '\n'.join([
            doc.properties['page_content'] for doc in retrieved_documents
        ])

        system_context = f"Papel: Você é um chatbot especializado em saúde mental que receberá um 'Contexto' com informações verídicas relacionadas à pergunta do usuário, que são provenientes de uma base de dados de fontes confiáveis. Você não é um profissional de saúde e não pode fornecer diagnósticos ou tratamentos, mas utiliza o Contexto para fornecer informações embasadas.\n\nContexto:{context}"  # noqa: E501
        messages = [
            ('system', system_context),
            ('human', f'Pergunta: {query}{context}\n\nAnswer:'),
        ]

        response = self.llm.generate_response(messages)
        return response


if __name__ == '__main__':
    from mental_helth_ai.rag.database.weaviate_impl import WeaviateImpl
    from mental_helth_ai.rag.llm.ollama_impl import OllamaLLM

    vector_db = WeaviateImpl()
    llm = OllamaLLM()
    rag_factory = RAGFactory(vector_db=vector_db, llm=llm)

    query = 'Responda em um parágrafo, o que é o Transtorno de Déficit de Atenção/Hiperatividade (TDAH)?'  # noqa: E501

    # Pergunta utilizando RAG
    response_rag = rag_factory.generate_response(query)
    print(f'Resposta RAG:\n{response_rag}')

    # Pergunta utilizando apenas o LLM
    response_llm = llm.generate_response(query)
    print(f'Resposta LLM:\n{response_llm}')
