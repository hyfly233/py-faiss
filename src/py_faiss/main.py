import logging
import os

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

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Document search API using FAISS and embeddings",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
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

def main() -> None:
    app_host = os.getenv('APP_HOST', '0.0.0.0')
    app_port = os.getenv('APP_PORT', 8080)

    uvicorn.run(
        "py_faiss.main:app",
        host=app_host,
        port=app_port,
        reload=True
    )
