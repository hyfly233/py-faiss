import time

from fastapi import APIRouter, Depends, HTTPException

from py_faiss.core.search_engine import SearchEngine
from py_faiss.dependencies import get_search_engine
from py_faiss.models.requests import SearchRequest, SearchResponse
from fastapi import APIRouter, Query
from typing import Optional, List
from pydantic import BaseModel

from py_faiss.services.document_service import get_document_service
router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    filter_doc_ids: Optional[List[str]] = None
    min_score: Optional[float] = 0.1

@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest, search_engine: SearchEngine = Depends(get_search_engine)):
    """搜索文档"""
    start_time = time.time()

    try:
        results = await search_engine.search(request.query, request.top_k)
        processing_time = time.time() - start_time

        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            processing_time=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@router.get("/search/stats")
async def get_search_stats(search_engine: SearchEngine = Depends(get_search_engine)):
    """获取搜索引擎统计信息"""
    try:
        stats = await search_engine.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.post("/")
async def search_documents(request: SearchRequest):
    """搜索文档"""
    document_service = await get_document_service()

    return await document_service.search_documents(
        query=request.query,
        top_k=request.top_k,
        filter_doc_ids=request.filter_doc_ids,
        min_score=request.min_score
    )