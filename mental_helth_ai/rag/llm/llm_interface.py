from abc import ABC, abstractmethod


class LLMInterface(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate a response from the LLM for the given prompt."""
        raise NotImplementedError
