from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8', extra='ignore'
    )

    EMBEDDING_MODEL: str
    EMBEDDING_DIMENSION: int
    FAISS_INDEX_PATH: str
    DOCUMENTS_PATH: str
    LLM_MODEL_NAME: str
    OPENAI_API_KEY: str
    WEAVIATE_URL: str = 'localhost'
    WEAVIATE_PORT: str = '8080'


settings = Settings()
