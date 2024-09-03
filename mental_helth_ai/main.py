from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mental_helth_ai.rag.database.weaviate_impl import WeaviateClient
from mental_helth_ai.rag.llm.openai_impl import OpenAILLM
from mental_helth_ai.rag.rag import RAGFactory

app = FastAPI()

vector_db = WeaviateClient()
llm = OpenAILLM()
rag_factory = RAGFactory(vector_db=vector_db, llm=llm)


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    response: str
    source_documents: list


@app.post('/rag/query', response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        response, source_documents = rag_factory.generate_response(
            request.query, request.top_k
        )

        return QueryResponse(
            response=response, source_documents=source_documents
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# TODO: Adicionar implementação para WebSocket
