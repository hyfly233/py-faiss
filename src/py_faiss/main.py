import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()


app = FastAPI(title=)


def main() -> None:
    app_host = os.getenv('APP_HOST', '0.0.0.0')
    app_port = os.getenv('APP_PORT', 8080)

    uvicorn.run(
        "py_faiss.main:app",
        host=app_host,
        port=app_port,
        reload=True
    )
