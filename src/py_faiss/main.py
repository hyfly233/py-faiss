import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import UJSONResponse

from py_faiss.api.v1.api import api_router
from py_faiss.config import settings
from py_faiss.core.search_engine import SearchEngine

load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppLifespan:
    def __init__(self):
        self.search_engine = None

    async def startup(self, app: FastAPI):
        """启动时初始化"""
        logger.info("Initializing application components...")

        # 初始化搜索引擎
        self.search_engine = SearchEngine()
        await self.search_engine.initialize()
        app.state.search_engine = self.search_engine

        # 初始化其他组件

        logger.info("Application startup complete")

    async def shutdown(self, app: FastAPI):
        """关闭时清理"""
        logger.info("Cleaning up application components...")

        if self.search_engine:
            await self.search_engine.cleanup()

        # 清理其他资源

        logger.info("Application shutdown complete")


# 创建全局实例
app_lifespan = AppLifespan()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期上下文管理器"""
    try:
        await app_lifespan.startup(app)
        yield
    finally:
        await app_lifespan.shutdown(app)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Document search API using FAISS and embeddings",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return UJSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# 包含路由
app.include_router(router=api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    return {
        "message": "Document Search API",
        "version": settings.VERSION,
        "docs": f"{settings.API_V1_STR}/docs"
    }

def main() -> None:
    app_host: str = os.getenv('APP_HOST', '0.0.0.0')
    app_port: int = int(os.getenv('APP_PORT', 8080))

    uvicorn.run(
        app="py_faiss.main:app",
        host=app_host,
        port=app_port,
        reload=True
    )


if __name__ == '__main__':
    main()
