from langchain_core.language_models.base import LanguageModelInput
from langchain_ollama import ChatOllama

from mental_helth_ai.settings import settings
from mental_helth_ai.rag.llm.llm_interface import LLMInterface


class OllamaLLM(LLMInterface):
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
