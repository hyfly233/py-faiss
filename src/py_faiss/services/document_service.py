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