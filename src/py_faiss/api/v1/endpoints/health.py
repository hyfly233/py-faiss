import asyncio
import logging
import psutil
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from py_faiss.core.embedding import get_embedding_service
from py_faiss.core.vector_store import get_vector_store
from py_faiss.services.document_service import get_document_service
from py_faiss.services.search_service import get_search_service
from py_faiss.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/test")
async def test():
    return [{"test": "health"}]
