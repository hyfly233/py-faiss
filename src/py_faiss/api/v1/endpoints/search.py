from fastapi import APIRouter, Depends

from py_faiss.core.search_engine import SearchEngine
from py_faiss.dependencies import get_search_engine
from py_faiss.models.requests import SearchRequest, SearchResponse

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest, search_engine: SearchEngine = Depends(get_search_engine)):
    pass


@router.get("/search/stats")
async def get_search_stats(search_engine: SearchEngine = Depends(get_search_engine)):
    pass
