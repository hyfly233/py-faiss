import time

from fastapi import APIRouter, Depends, HTTPException

from py_faiss.core.search_engine import SearchEngine
from py_faiss.dependencies import get_search_engine
from py_faiss.models.requests import SearchRequest, SearchResponse

router = APIRouter()

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
