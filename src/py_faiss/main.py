import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import UJSONResponse

from py_faiss.config import settings

load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("Starting up Document Search API...")

    try:
        # 初始化搜索引擎
        # search_engine = SearchEngine()
        # await search_engine.initialize()
        #
        # # 将搜索引擎存储在应用状态中
        # app.state.search_engine = search_engine
        logger.info("Search engine initialized successfully")

        yield  # 应用运行期间

    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}")
        raise
    finally:
        # 关闭时执行
        logger.info("Shutting down Document Search API...")
        if hasattr(app.state, 'search_engine'):
            await app.state.search_engine.cleanup()
        logger.info("Application shutdown complete")

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
# app.include_router(api_router, prefix=settings.API_V1_STR)


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
