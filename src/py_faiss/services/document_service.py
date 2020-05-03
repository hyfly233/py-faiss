import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import uuid
from datetime import datetime
import hashlib
import os

from py_faiss.core.document_processor import document_processor
from py_faiss.core.embedding import get_embedding_service
from py_faiss.core.vector_store import get_vector_store, Document
from py_faiss.config import settings

logger = logging.getLogger(__name__)


class DocumentService:
    """文档服务 - 提供完整的文档管理功能"""

    def __init__(self):
        self.document_processor = document_processor
        self.embedding_service = None
        self.vector_store = None

        # 文档状态
        self.processing_status: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        """初始化服务"""
        try:
            self.embedding_service = await get_embedding_service()
            self.vector_store = await get_vector_store()
            logger.info("文档服务初始化完成")
        except Exception as e:
            logger.error(f"文档服务初始化失败: {e}")
            raise