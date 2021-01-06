import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import aiofiles
import pytest

from py_faiss.core.document_processor import DocumentProcessor, document_processor

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

    @pytest.mark.asyncio
    async def test_save_temp_file(self, processor, temp_dir):
        """测试保存临时文件"""
        content = f"测试文件内容"
        filename = "test.txt"

        # 临时修改处理器的临时目录
        original_temp_dir = processor.temp_dir
        processor.temp_dir = temp_dir

        try:
            file_path = await processor.save_temp_file(content, filename)

            assert file_path.exists()
            assert file_path.name.endswith("_test.txt")
            assert file_path.parent == temp_dir

            # 验证文件内容
            async with aiofiles.open(file_path, 'rb') as f:
                saved_content = await f.read()
                assert saved_content == content

        finally:
            processor.temp_dir = original_temp_dir
