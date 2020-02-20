import logging
from pathlib import Path
from typing import Union

from py_faiss.config import settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """文档处理器 - 支持多种文档格式的文本提取和处理"""

    def __init__(self):
        self.supported_extensions = {
            '.docx', '.doc',  # Word 文档
            '.pdf',  # PDF 文档
            '.txt', '.md',  # 文本文档
            '.xlsx', '.xls',  # Excel 文档
            '.csv',  # CSV 文档
            '.json',  # JSON 文档
            '.xml',  # XML 文档
        }

        # 文本分割配置
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

        # 临时文件目录
        self.temp_dir = Path(settings.TEMP_PATH)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def extract_text(self, file_path: Union[str, Path]):
        """
        从文档中提取文本

        Args:
            file_path: 文档文件路径

        Returns:
            提取的文本内容
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 检查文件大小
        file_size = file_path.stat().st_size
        if file_size > settings.MAX_FILE_SIZE:
            raise ValueError(f"文件过大: {file_size / 1024 / 1024:.1f}MB > {settings.MAX_FILE_SIZE / 1024 / 1024}MB")

