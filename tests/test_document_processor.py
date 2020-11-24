import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import aiofiles

from py_faiss.core.document_processor import DocumentProcessor, document_processor
from py_faiss.config import settings


class TestDocumentProcessor:
    """文档处理器测试类"""

    @pytest.fixture
    def processor(self):
        """创建文档处理器实例"""
        return DocumentProcessor()

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def sample_texts(self):
        """示例文本内容"""
        return {
            'simple': "这是一个简单的测试文档。",
            'long': "这是一个很长的文档。" + "测试内容。" * 100,
            'multi_paragraph': """第一段内容。

第二段内容，包含更多信息。

第三段内容。""",
            'unicode': "这是包含中文、English和éñglish的文档。",
            'empty': "",
            'whitespace': "   \n\t   \n   "
        }

    # ========== 基础功能测试 ==========

    def test_get_supported_types(self, processor):
        """测试获取支持的文件类型"""
        supported_types = processor.get_supported_types()

        assert isinstance(supported_types, list)
        assert len(supported_types) > 0
        assert '.txt' in supported_types
        assert '.pdf' in supported_types
        assert '.docx' in supported_types