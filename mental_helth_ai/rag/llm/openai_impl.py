import os

from langchain.llms import OpenAI

from mental_helth_ai.rag.llm.llm_interface import LLMInterface
from mental_helth_ai.settings import Settings

settings = Settings()


class OpenAIModel(LLMInterface):
    def __init__(
        self,
        model_name: str = Settings().LLM_MODEL_NAME,
        use_auth_token: bool = True,
    ):
        self.model_name = model_name
        self.api_key = os.getenv('OPENAI_API_KEY', None)
        if use_auth_token and not self.api_key:
            raise ValueError('API key for OpenAI is required!')
        self.llm = OpenAI(
            model_name=self.model_name, openai_api_key=self.api_key
        )

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.llm(prompt)
            return response
        except Exception as e:
            print(f'Error generating response: {e}')
            return 'Desculpe, ocorreu um erro ao gerar a resposta.'
