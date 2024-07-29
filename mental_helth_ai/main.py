from fastapi import Depends, FastAPI

from mental_helth_ai.dependencies.get_db import get_db
from mental_helth_ai.rag.database.faiss_db_impl import FAISSDatabase

app = FastAPI()


# TODO: Corrigir get_db e rota
@app.get('/search')
def search(query: str, db: FAISSDatabase = Depends(get_db)):
    try:
        results = db.search(query)
        return results
    except RuntimeError as e:
        return {'error': str(e)}
