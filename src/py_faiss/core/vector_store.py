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