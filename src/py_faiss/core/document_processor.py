import asyncio
import logging
from pathlib import Path
from typing import Union

from docx import Document

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

        # 获取文件扩展名
        extension = file_path.suffix.lower()

        if extension not in self.supported_extensions:
            raise ValueError(f"不支持的文件格式: {extension}")

        try:
            # 根据文件类型选择提取方法
            if extension in ['.docx', '.doc']:
                text = await self._extract_from_docx(file_path)
            elif extension == '.pdf':
                text = await self._extract_from_pdf(file_path)
            elif extension in ['.txt', '.md']:
                text = await self._extract_from_text(file_path)
            elif extension in ['.xlsx', '.xls']:
                text = await self._extract_from_excel(file_path)
            elif extension == '.csv':
                text = await self._extract_from_csv(file_path)
            elif extension == '.json':
                text = await self._extract_from_json(file_path)
            elif extension == '.xml':
                text = await self._extract_from_xml(file_path)
            else:
                raise ValueError(f"暂不支持的文件格式: {extension}")

            logger.info(f"成功提取文本: {file_path.name}, 长度: {len(text)} 字符")
            return text

        except Exception as e:
            logger.error(f"文本提取失败 {file_path.name}: {e}")
            raise

    async def _extract_from_docx(self, file_path: Path) -> str:
        """从 DOCX 文件提取文本"""
        try:
            # 在线程池中执行 IO 密集型操作
            loop = asyncio.get_event_loop()

            def _read_docx():
                doc = Document(file_path)
                paragraphs = []

                # 提取段落文件
                for paragraph in doc.paragraphs:
                    text = paragraph.text.strip()
                    if text:
                        paragraphs.append(text)

                # 提取表格文本
                for table in doc.tables:
                    for row in table.rows:
                        row_text = []
                        for cell in row.cells:
                            cell_text = cell.text.strip()
                            if cell_text:
                                row_text.append(cell_text)
                        if row_text:
                            paragraphs.append(' | '.join(row_text))

                return '\n'.join(paragraphs)

            text = await loop.run_in_executor(None, _read_docx)
            return text

        except Exception as e:
            raise Exception(f"DOCX 处理失败: {e}")

    async def _extract_from_pdf(self, file_path: Path) -> str:
        pass

    async def _extract_from_text(self, file_path: Path) -> str:
        pass

    async def _extract_from_excel(self, file_path: Path) -> str:
        pass

    async def _extract_from_csv(self, file_path: Path) -> str:
        pass

    async def _extract_from_json(self, file_path: Path) -> str:
        pass

    async def _extract_from_xml(self, file_path: Path) -> str:
        pass