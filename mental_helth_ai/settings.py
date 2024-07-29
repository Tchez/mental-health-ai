from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8', extra='ignore'
    )

    EMBEDDING_MODEL: str
    EMBEDDING_DIMENSION: int
    FAISS_INDEX_PATH: str
    DOCUMENTS_PATH: str


settings = Settings()
