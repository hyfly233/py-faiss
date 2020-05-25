from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from typing import List, Optional
import logging

from py_faiss.services.document_service import get_document_service

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get("/test")
async def test():
    return [{"test": "documents"}]
