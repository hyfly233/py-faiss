import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import re
import json
from collections import defaultdict
import numpy as np

from py_faiss.core.embedding import get_embedding_service
from py_faiss.core.vector_store import get_vector_store, SearchResult
from py_faiss.services.document_service import get_document_service
from py_faiss.config import settings

logger = logging.getLogger(__name__)

class SearchFilter:
    """搜索过滤器"""
    def __init__(
        self,
        doc_ids: Optional[List[str]] = None,
        file_names: Optional[List[str]] = None,
        file_types: Optional[List[str]] = None,
        date_range: Optional[Tuple[str, str]] = None,
        min_score: float = 0.0,
        metadata_filters: Optional[Dict[str, Any]] = None
    ):
        self.doc_ids = doc_ids
        self.file_names = file_names
        self.file_types = file_types
        self.date_range = date_range
        self.min_score = min_score
        self.metadata_filters = metadata_filters or {}

class SearchOptions:
    """搜索选项"""
    def __init__(
        self,
        search_type: str = "vector",  # vector, hybrid, keyword
        top_k: int = 10,
        enable_rerank: bool = False,
        enable_highlight: bool = True,
        enable_summary: bool = False,
        chunk_merge: bool = True,
        diversity_threshold: float = 0.7
    ):
        self.search_type = search_type
        self.top_k = top_k
        self.enable_rerank = enable_rerank
        self.enable_highlight = enable_highlight
        self.enable_summary = enable_summary
        self.chunk_merge = chunk_merge
        self.diversity_threshold = diversity_threshold

class EnhancedSearchResult:
    """增强的搜索结果"""
    def __init__(
        self,
        doc_id: str,
        file_name: str,
        file_path: str,
        chunks: List[Dict[str, Any]],
        max_score: float,
        avg_score: float,
        rank: int,
        highlighted_text: str = "",
        summary: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.doc_id = doc_id
        self.file_name = file_name
        self.file_path = file_path
        self.chunks = chunks
        self.max_score = max_score
        self.avg_score = avg_score
        self.rank = rank
        self.highlighted_text = highlighted_text
        self.summary = summary
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'doc_id': self.doc_id,
            'file_name': self.file_name,
            'file_path': self.file_path,
            'chunks': self.chunks,
            'max_score': float(self.max_score),
            'avg_score': float(self.avg_score),
            'rank': self.rank,
            'highlighted_text': self.highlighted_text,
            'summary': self.summary,
            'metadata': self.metadata,
            'chunk_count': len(self.chunks)
        }