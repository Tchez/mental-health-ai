from langchain_core.language_models.base import LanguageModelInput
from langchain_ollama import ChatOllama

from mental_health_ai.rag.llm.llm_interface import LLMInterface
from mental_health_ai.settings import settings


class OllamaLLM(LLMInterface):
    """Implementation of the LLMInterface using the Ollama language model.

    Attributes:
        model_name (str): The name of the language model to use.

    Examples:
        >>> llm = OllamaLLM()
        >>> messages = [('human', 'Olá, como vai?')]
        >>> response = llm.generate_response(messages)
        >>> print(response)
        "Olá, estou bem, obrigado. Como posso ajud
    """  # noqa: E501

    def __init__(
        self,
        model_name: str = settings.LLM_MODEL_NAME,
    ):
        self.model_name = model_name
        self.llm = ChatOllama(model=self.model_name)

    def generate_response(self, messages: LanguageModelInput) -> str:
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f'Error generating response: {e}')
            return 'Desculpe, ocorreu um erro ao gerar a resposta.'
