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

    def test_is_supported_file(self, processor):
        """测试文件类型支持检查"""
        # 支持的文件类型
        assert processor.is_supported_file("test.txt") == True
        assert processor.is_supported_file("document.pdf") == True
        assert processor.is_supported_file("report.docx") == True
        assert processor.is_supported_file("data.csv") == True

        # 不支持的文件类型
        assert processor.is_supported_file("image.jpg") == False
        assert processor.is_supported_file("video.mp4") == False
        assert processor.is_supported_file("archive.zip") == False

        # 边界情况
        assert processor.is_supported_file("") == False
        assert processor.is_supported_file("noextension") == False
        assert processor.is_supported_file(".txt") == True

    def test_split_text_by_sentences(self, processor, sample_texts):
        """测试句子分割"""
        # 简单文本
        result = processor._split_text_by_sentences(sample_texts['simple'])
        assert len(result) == 1
        assert result[0] == "这是一个简单的测试文档。"

        # 多段落文本
        result = processor._split_text_by_sentences(sample_texts['multi_paragraph'])
        assert len(result) >= 3

        # 空文本
        result = processor._split_text_by_sentences(sample_texts['empty'])
        assert len(result) == 0

        # 纯空白
        result = processor._split_text_by_sentences(sample_texts['whitespace'])
        assert len(result) == 0

    def test_chunk_text(self, processor, sample_texts):
        """测试文本分块"""
        # 短文本 - 应该返回一个块
        chunks = processor._chunk_text(sample_texts['simple'])
        assert len(chunks) == 1
        assert chunks[0] == sample_texts['simple']

        # 长文本 - 应该分成多个块
        chunks = processor._chunk_text(sample_texts['long'])
        assert len(chunks) > 1

        # 验证每个块的长度不超过限制
        for chunk in chunks:
            assert len(chunk) <= processor.chunk_size

        # 空文本
        chunks = processor._chunk_text(sample_texts['empty'])
        assert len(chunks) == 0

        # 自定义参数测试
        small_chunks = processor._chunk_text(sample_texts['long'], chunk_size=100, overlap=20)
        assert len(small_chunks) > len(chunks)  # 更小的块应该产生更多分片

    # ========== 文件创建和保存测试 ==========

    @pytest.mark.asyncio
    async def test_save_temp_file(self, processor, temp_dir):
        """测试保存临时文件"""
        content = b"测试文件内容"
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

    @pytest.mark.asyncio
    async def test_save_temp_file_with_special_chars(self, processor, temp_dir):
        """测试保存包含特殊字符的文件名"""
        content = b"content"
        filename = "测试文件 (1) [copy].txt"

        processor.temp_dir = temp_dir

        file_path = await processor.save_temp_file(content, filename)

        assert file_path.exists()
        # 文件名应该被清理但保持可读性
        assert "txt" in file_path.name

    # ========== TXT 文件处理测试 ==========

    def create_temp_txt_file(self, temp_dir: Path, content: str, filename: str = "test.txt") -> Path:
        """创建临时TXT文件"""
        file_path = temp_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path

    @pytest.mark.asyncio
    async def test_process_txt_file(self, processor, temp_dir, sample_texts):
        """测试TXT文件处理"""
        file_path = self.create_temp_txt_file(temp_dir, sample_texts['multi_paragraph'])

        result = await processor._process_txt_file(file_path)

        assert result['status'] == 'success'
        assert len(result['chunks']) >= 3
        assert result['file_name'] == file_path.name
        assert result['file_size'] > 0
        assert 'processing_time' in result
        assert 'document_hash' in result