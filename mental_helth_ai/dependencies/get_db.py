from typing import Generator

from mental_helth_ai.rag.database.faiss_db_impl import FAISSDatabase
from mental_helth_ai.settings import Settings

settings = Settings()


def get_db() -> Generator[FAISSDatabase, None, None]:
    try:
        db = FAISSDatabase(
            model_name=settings.EMBEDDING_MODEL,
            index_path=settings.FAISS_INDEX_PATH,
        )
        db.load_index()
        yield db
        db.save_index()
    except Exception as e:
        print(f'Configuration error: {e}')
        yield None
