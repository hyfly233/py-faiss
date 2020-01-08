from fastapi import APIRouter

from src.py_faiss.api.v1.endpoints import documents, search, health

api_router = APIRouter()

# 包含文档相关路由
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])

# 包含搜索相关路由
api_router.include_router(search.router, prefix="/search", tags=["search"])

# 健康检查路由
api_router.include_router(health.router, prefix="/health", tags=["health"])
