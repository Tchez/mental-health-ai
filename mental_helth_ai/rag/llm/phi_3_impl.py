import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mental_helth_ai.rag.llm.llm_interface import LLMInterface
from mental_helth_ai.settings import Settings

settings = Settings()


class PhiModel(LLMInterface):
    def __init__(
        self,
        model_name: str = settings.LLM_MODEL_NAME,
    ):
        self.model_name = model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        except EnvironmentError as e:
            print(f'Error loading model or tokenizer: {e}')
            raise e
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)

    def generate_response(self, prompt: str) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors='pt').to(
                self.device
            )
            outputs = self.model.generate(**inputs, max_length=512)
            response = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            return response
        except Exception as e:
            print(f'Error generating response: {e}')
            return 'Desculpe, ocorreu um erro ao gerar a resposta.'
