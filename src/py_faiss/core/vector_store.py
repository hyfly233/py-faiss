import asyncio
import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import faiss
from datetime import datetime
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
import threading

from py_faiss.config import settings

logger = logging.getLogger(__name__)

class Document:
    """文档元数据类"""

    def __init__(
            self,
            doc_id: str,
            file_path: str,
            file_name: str,
            chunk_index: int,
            text: str,
            embedding: Optional[np.ndarray] = None,
            metadata: Optional[Dict[str, Any]] = None
    ):
        self.doc_id = doc_id
        self.file_path = file_path
        self.file_name = file_name
        self.chunk_index = chunk_index
        self.text = text
        self.embedding = embedding
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'doc_id': self.doc_id,
            'file_path': self.file_path,
            'file_name': self.file_name,
            'chunk_index': self.chunk_index,
            'text': self.text,
            'metadata': self.metadata,
            'created_at': self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """从字典创建"""
        doc = cls(
            doc_id=data['doc_id'],
            file_path=data['file_path'],
            file_name=data['file_name'],
            chunk_index=data['chunk_index'],
            text=data['text'],
            metadata=data.get('metadata', {})
        )
        doc.created_at = data.get('created_at', datetime.now().isoformat())
        return doc

class SearchResult:
    """搜索结果类"""

    def __init__(self, document: Document, score: float, rank: int):
        self.document = document
        self.score = score
        self.rank = rank

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'doc_id': self.document.doc_id,
            'file_name': self.document.file_name,
            'file_path': self.document.file_path,
            'chunk_index': self.document.chunk_index,
            'text': self.document.text,
            'score': float(self.score),
            'rank': self.rank,
            'metadata': self.document.metadata,
            'created_at': self.document.created_at
        }

class VectorStore:
    """FAISS 向量存储实现"""

    def __init__(
            self,
            dimension: int = None,
            index_type: str = "IndexFlatIP",  # 内积索引，适合归一化向量
            storage_path: str = None
    ):
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.index_type = index_type
        self.storage_path = Path(storage_path or settings.INDEX_PATH)

        # 确保存储目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 索引和元数据
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []
        self.doc_id_to_idx: Dict[str, List[int]] = {}  # doc_id -> [indices]
        self.idx_to_doc_idx: Dict[int, int] = {}  # faiss_index -> document_index

        # 文件路径
        self.index_file = self.storage_path / "faiss_index.bin"
        self.metadata_file = self.storage_path / "metadata.pkl"
        self.config_file = self.storage_path / "config.json"

        # 线程安全
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)

        # 统计信息
        self._stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'index_size': 0,
            'created_at': None,
            'last_updated': None
        }

    async def initialize(self) -> bool:
        """初始化向量存储"""
        try:
            with self._lock:
                # 尝试加载现有索引
                if await self._load_existing_index():
                    logger.info(f"加载现有索引成功: {len(self.documents)} 个文档")
                else:
                    # 创建新索引
                    await self._create_new_index()
                    logger.info(f"创建新索引成功: 维度 {self.dimension}")

                # 更新统计信息
                await self._update_stats()

            return True

        except Exception as e:
            logger.error(f"向量存储初始化失败: {e}")
            raise

    async def _create_new_index(self):
        """创建新的 FAISS 索引"""
        loop = asyncio.get_event_loop()

        def _create():
            if self.index_type == "IndexFlatIP":
                # 内积索引，适合归一化向量的余弦相似度
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "IndexFlatL2":
                # L2 距离索引
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "IndexIVFFlat":
                # 倒排文件索引，适合大量数据
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            elif self.index_type == "IndexHNSW":
                # HNSW 索引，平衡速度和精度
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                # 默认使用 IndexFlatIP
                self.index = faiss.IndexFlatIP(self.dimension)

            logger.info(f"创建索引类型: {type(self.index).__name__}")

        await loop.run_in_executor(self._executor, _create)