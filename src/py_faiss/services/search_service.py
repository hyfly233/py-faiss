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