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

    async def _process_document_async(
        self,
        doc_id: str,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """异步处理文档"""
        try:
            # 更新状态：文本提取
            self._update_processing_status(doc_id, 10, "正在提取文档文本...")

            # 处理文档
            process_result = await self.document_processor.process_document(file_path)

            if process_result['status'] != 'success':
                raise Exception(f"文档处理失败: {process_result.get('error', 'Unknown error')}")

            chunks = process_result['chunks']

            # 更新状态：生成嵌入向量
            self._update_processing_status(doc_id, 30, f"正在生成嵌入向量... ({len(chunks)} 个文本块)")

            # 生成嵌入向量
            embeddings = await self.embedding_service.get_embeddings_batch(
                chunks,
                show_progress=False
            )

            # 更新状态：创建文档对象
            self._update_processing_status(doc_id, 70, "正在创建文档索引...")

            # 创建文档对象
            documents = []
            file_hash = hashlib.md5(str(file_path).encode()).hexdigest()

            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    'file_size': process_result['file_size'],
                    'processing_time': process_result['processing_time'],
                    'document_hash': process_result['document_hash'],
                    'file_hash': file_hash,
                    'chunk_length': len(chunk),
                    **(metadata or {})
                }

                doc = Document(
                    doc_id=doc_id,
                    file_path=str(file_path),
                    file_name=process_result['file_name'],
                    chunk_index=i,
                    text=chunk,
                    metadata=doc_metadata
                )
                documents.append(doc)

            # 更新状态：添加到向量存储
            self._update_processing_status(doc_id, 90, "正在添加到向量数据库...")

            # 添加到向量存储
            success = await self.vector_store.add_documents(documents, embeddings)

            if not success:
                raise Exception("添加到向量存储失败")

            # 完成处理
            self.processing_status[doc_id] = {
                'status': 'completed',
                'filename': process_result['file_name'],
                'started_at': self.processing_status[doc_id]['started_at'],
                'completed_at': datetime.now().isoformat(),
                'progress': 100,
                'message': '文档处理完成',
                'chunks_count': len(chunks),
                'file_size': process_result['file_size'],
                'document_hash': process_result['document_hash']
            }

            logger.info(f"文档处理完成: {doc_id} ({process_result['file_name']})")

        except Exception as e:
            logger.error(f"文档处理失败 {doc_id}: {e}")
            self.processing_status[doc_id] = {
                'status': 'error',
                'error': str(e),
                'filename': self.processing_status[doc_id].get('filename', 'Unknown'),
                'started_at': self.processing_status[doc_id]['started_at'],
                'failed_at': datetime.now().isoformat(),
                'message': f'文档处理失败: {str(e)}'
            }
        finally:
            # 清理临时文件
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                logger.warning(f"清理临时文件失败: {e}")

    def _update_processing_status(self, doc_id: str, progress: int, message: str):
        """更新处理状态"""
        if doc_id in self.processing_status:
            self.processing_status[doc_id].update({
                'progress': progress,
                'message': message,
                'updated_at': datetime.now().isoformat()
            })

