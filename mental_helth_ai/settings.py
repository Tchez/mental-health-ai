from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8', extra='ignore'
    )

    IS_LOCAL_EMBEDDING: bool = False
    OPENAI_API_KEY: str = ''
    LLM_MODEL_NAME: str
    WEAVIATE_URL: str = 'localhost'
    WEAVIATE_PORT: str = '8080'


settings = Settings()
