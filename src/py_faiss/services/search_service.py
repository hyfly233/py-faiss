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
