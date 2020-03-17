import os
from datetime import datetime
from typing import List, Dict, Any

import faiss

from py_faiss.config import settings
from py_faiss.core.document_processor import DocumentProcessor
from py_faiss.core.embedding import EmbeddingService
from py_faiss.models.requests import SearchResult


class SearchEngine:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.document_processor = DocumentProcessor()
        self.index = None
        self.documents = []
        self.chunks = []
        self.index_file = os.path.join(settings.INDEX_PATH, "faiss_index.bin")
        self.metadata_file = os.path.join(settings.INDEX_PATH, "metadata.pkl")

    async def initialize(self):
        """初始化搜索引擎"""
        await self.embedding_service.initialize()

        # 创建或加载索引
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            await self.load_index()
        else:
            self.index = faiss.IndexFlatIP(settings.EMBEDDING_DIMENSION)

    async def add_document(self, file_path: str, document_id: str) -> Dict[str, Any]:
        """添加文档到索引"""
        try:
            # 处理文档
            text_content = await self.document_processor.extract_text(file_path)
            if not text_content:
                return {"status": "error", "message": "无法提取文档内容"}

            # 分割文本
            chunks = self.document_processor.split_text(text_content)

            # 生成嵌入向量
            embeddings = await self.embedding_service.get_embeddings_batch(chunks)

            # 标准化向量
            faiss.normalize_L2(embeddings)

            # 添加到索引
            self.index.add(embeddings.astype('float32'))

            # 存储文档信息
            filename = os.path.basename(file_path)
            for i, chunk in enumerate(chunks):
                self.documents.append({
                    'document_id': document_id,
                    'file_path': file_path,
                    'file_name': filename,
                    'chunk_index': i,
                    'text': chunk,
                    'created_at': datetime.now().isoformat()
                })
                self.chunks.append(chunk)

            # 保存索引
            await self.save_index()

            return {
                "status": "success",
                "chunks_count": len(chunks),
                "message": f"成功添加 {len(chunks)} 个文本块"
            }

        except Exception as e:
            return {"status": "error", "message": f"处理文档失败: {str(e)}"}

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        pass

    async def save_index(self):
        """保存索引"""
        try:
            pass
        except Exception as e:
            raise Exception(f"保存索引失败: {str(e)}")

    async def load_index(self):
        """加载索引"""
        try:
            pass
        except Exception as e:
            raise Exception(f"加载索引失败: {str(e)}")

    async def get_stats(self) -> Dict[str, Any]:
        """获取搜索引擎统计信息"""
        return {
            "total_documents": len(set(doc['document_id'] for doc in self.documents)),
            "total_chunks": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_model": settings.EMBEDDING_MODEL
        }

    async def cleanup(self):
        """清理资源"""
        await self.embedding_service.cleanup()
