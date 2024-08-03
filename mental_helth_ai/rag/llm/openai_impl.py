from langchain_core.language_models.base import LanguageModelInput
from langchain_openai import ChatOpenAI

from mental_helth_ai.rag.llm.llm_interface import LLMInterface
from mental_helth_ai.settings import settings


class OpenAILLM(LLMInterface):
    """Implementation of the LLMInterface using the OpenAI language model.

    Attributes:
        model_name (str): The name of the language model to use.

    Examples:
        >>> llm = OpenAIModel()
        >>> messages = [('human', 'Olá, como vai?')]
        >>> response = llm.generate_response(messages)
        >>> print(response)
        "Olá, estou bem, obrigado. Como posso ajudar?
    """  # noqa: E501

    def __init__(
        self,
        model_name: str = settings.LLM_MODEL_NAME,
        use_auth_token: bool = True,
    ):
        self.model_name = model_name
        self.api_key = settings.OPENAI_API_KEY
        if use_auth_token and not self.api_key:
            raise ValueError('API key for OpenAI is required!')
        self.llm = ChatOpenAI(model=self.model_name, api_key=self.api_key)

    def generate_response(self, messages: LanguageModelInput) -> str:
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f'Error generating response: {e}')
            return 'Desculpe, ocorreu um erro ao gerar a resposta.'
