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

    async def upload_and_process_document(
        self,
        file_content: bytes,
        filename: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        上传并处理文档

        Args:
            file_content: 文件内容
            filename: 文件名
            user_id: 用户ID
            metadata: 额外元数据

        Returns:
            处理结果
        """
        # 生成文档ID
        doc_id = str(uuid.uuid4())

        try:
            # 保存临时文件
            temp_file_path = await self.document_processor.save_temp_file(file_content, filename)

            # 初始化处理状态
            self.processing_status[doc_id] = {
                'status': 'processing',
                'filename': filename,
                'user_id': user_id,
                'started_at': datetime.now().isoformat(),
                'progress': 0,
                'message': '开始处理文档...'
            }

            # 异步处理文档
            asyncio.create_task(self._process_document_async(doc_id, temp_file_path, metadata))

            return {
                'doc_id': doc_id,
                'status': 'processing',
                'message': '文档上传成功，正在处理中...',
                'filename': filename
            }

        except Exception as e:
            logger.error(f"上传文档失败: {e}")
            self.processing_status[doc_id] = {
                'status': 'error',
                'error': str(e),
                'filename': filename,
                'failed_at': datetime.now().isoformat()
            }
            return {
                'doc_id': doc_id,
                'status': 'error',
                'error': str(e),
                'message': '文档上传失败'
            }
