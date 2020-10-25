import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import aiofiles

from py_faiss.core.document_processor import DocumentProcessor, document_processor
from py_faiss.config import settings
