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

class SearchOptions:
    """搜索选项"""
    def __init__(
        self,
        search_type: str = "vector",  # vector, hybrid, keyword
        top_k: int = 10,
        enable_rerank: bool = False,
        enable_highlight: bool = True,
        enable_summary: bool = False,
        chunk_merge: bool = True,
        diversity_threshold: float = 0.7
    ):
        self.search_type = search_type
        self.top_k = top_k
        self.enable_rerank = enable_rerank
        self.enable_highlight = enable_highlight
        self.enable_summary = enable_summary
        self.chunk_merge = chunk_merge
        self.diversity_threshold = diversity_threshold

class EnhancedSearchResult:
    """增强的搜索结果"""
    def __init__(
        self,
        doc_id: str,
        file_name: str,
        file_path: str,
        chunks: List[Dict[str, Any]],
        max_score: float,
        avg_score: float,
        rank: int,
        highlighted_text: str = "",
        summary: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.doc_id = doc_id
        self.file_name = file_name
        self.file_path = file_path
        self.chunks = chunks
        self.max_score = max_score
        self.avg_score = avg_score
        self.rank = rank
        self.highlighted_text = highlighted_text
        self.summary = summary
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'doc_id': self.doc_id,
            'file_name': self.file_name,
            'file_path': self.file_path,
            'chunks': self.chunks,
            'max_score': float(self.max_score),
            'avg_score': float(self.avg_score),
            'rank': self.rank,
            'highlighted_text': self.highlighted_text,
            'summary': self.summary,
            'metadata': self.metadata,
            'chunk_count': len(self.chunks)
        }


class SearchService:
    """搜索服务 - 提供高级搜索功能"""

    def __init__(self):
        self.embedding_service = None
        self.vector_store = None
        self.document_service = None

        # 搜索历史和缓存
        self.search_history: List[Dict[str, Any]] = []
        self.search_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5分钟缓存

        # 搜索统计
        self.search_stats = {
            'total_searches': 0,
            'avg_search_time': 0.0,
            'popular_queries': defaultdict(int),
            'search_types': defaultdict(int)
        }

    async def initialize(self):
        """初始化搜索服务"""
        try:
            self.embedding_service = await get_embedding_service()
            self.vector_store = await get_vector_store()
            self.document_service = await get_document_service()
            logger.info("搜索服务初始化完成")
        except Exception as e:
            logger.error(f"搜索服务初始化失败: {e}")
            raise

    async def search(
            self,
            query: str,
            options: SearchOptions = None,
            filters: SearchFilter = None,
            user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        统一搜索入口

        Args:
            query: 搜索查询
            options: 搜索选项
            filters: 搜索过滤器
            user_id: 用户ID

        Returns:
            搜索结果
        """
        if not query or not query.strip():
            return self._empty_search_result(query, "查询不能为空")

        options = options or SearchOptions()
        filters = filters or SearchFilter()

        start_time = datetime.now()

        try:
            # 检查缓存
            cache_key = self._generate_cache_key(query, options, filters)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.info(f"返回缓存结果: {query}")
                return cached_result

            # 根据搜索类型调用不同的搜索方法
            if options.search_type == "vector":
                results = await self._vector_search(query, options, filters)
            elif options.search_type == "hybrid":
                results = await self._hybrid_search(query, options, filters)
            elif options.search_type == "keyword":
                results = await self._keyword_search(query, options, filters)
            else:
                results = await self._vector_search(query, options, filters)

            # 后处理
            if options.chunk_merge:
                results = await self._merge_chunks(results)

            if options.enable_rerank:
                results = await self._rerank_results(results, query)

            if options.enable_highlight:
                results = await self._add_highlights(results, query)

            if options.enable_summary:
                results = await self._add_summaries(results, query)

            # 多样性过滤
            results = await self._apply_diversity_filter(results, options.diversity_threshold)

            # 计算搜索时间
            search_time = (datetime.now() - start_time).total_seconds()

            # 构建最终结果
            final_result = {
                'query': query,
                'results': [result.to_dict() for result in results[:options.top_k]],
                'total_results': len(results),
                'search_time': search_time,
                'search_type': options.search_type,
                'timestamp': datetime.now().isoformat(),
                'options': {
                    'top_k': options.top_k,
                    'search_type': options.search_type,
                    'enable_rerank': options.enable_rerank,
                    'enable_highlight': options.enable_highlight,
                    'chunk_merge': options.chunk_merge
                }
            }

            # 缓存结果
            self._save_to_cache(cache_key, final_result)

            # 记录搜索历史和统计
            await self._record_search(query, options, search_time, len(results), user_id)

            return final_result

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return self._empty_search_result(query, f"搜索失败: {str(e)}")

    async def _vector_search(
            self,
            query: str,
            options: SearchOptions,
            filters: SearchFilter
    ) -> List[EnhancedSearchResult]:
        """向量搜索"""
        try:
            # 生成查询向量
            query_embedding = await self.embedding_service.get_embedding(query)

            # 执行向量搜索
            search_results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=options.top_k * 3,  # 搜索更多结果用于后处理
                filter_doc_ids=filters.doc_ids,
                min_score=filters.min_score
            )

            # 应用其他过滤器
            filtered_results = await self._apply_filters(search_results, filters)

            # 转换为增强结果
            enhanced_results = await self._convert_to_enhanced_results(filtered_results)

            return enhanced_results

        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []

    async def _hybrid_search(
            self,
            query: str,
            options: SearchOptions,
            filters: SearchFilter
    ) -> List[EnhancedSearchResult]:
        """混合搜索（向量 + 关键词）"""
        try:
            # 向量搜索
            vector_results = await self._vector_search(query, options, filters)

            # 关键词搜索
            keyword_results = await self._keyword_search(query, options, filters)

            # 合并和重新排序
            combined_results = await self._combine_search_results(
                vector_results, keyword_results, query
            )

            return combined_results

        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            return []

    async def _keyword_search(
            self,
            query: str,
            options: SearchOptions,
            filters: SearchFilter
    ) -> List[EnhancedSearchResult]:
        """关键词搜索"""
        try:
            # 获取所有文档
            all_documents = await self.vector_store.list_documents()

            # 提取关键词
            keywords = self._extract_keywords(query)

            # 搜索匹配的文档
            matching_results = []

            for doc_info in all_documents:
                doc_id = doc_info['doc_id']

                # 应用过滤器
                if filters.doc_ids and doc_id not in filters.doc_ids:
                    continue

                # 获取文档详情
                doc_details = await self.document_service.get_document_details(doc_id)
                if not doc_details:
                    continue

                # 计算关键词匹配分数
                doc_score = self._calculate_keyword_score(doc_details, keywords)

                if doc_score > filters.min_score:
                    # 找到最佳匹配的块
                    best_chunks = self._find_best_matching_chunks(
                        doc_details['chunks'], keywords, options.top_k
                    )

                    if best_chunks:
                        result = EnhancedSearchResult(
                            doc_id=doc_id,
                            file_name=doc_details['file_name'],
                            file_path=doc_details['file_path'],
                            chunks=best_chunks,
                            max_score=max(chunk['score'] for chunk in best_chunks),
                            avg_score=sum(chunk['score'] for chunk in best_chunks) / len(best_chunks),
                            rank=len(matching_results),
                            metadata=doc_details['metadata']
                        )
                        matching_results.append(result)

            # 按分数排序
            matching_results.sort(key=lambda x: x.max_score, reverse=True)

            return matching_results

        except Exception as e:
            logger.error(f"关键词搜索失败: {e}")
            return []

    async def _apply_filters(
            self,
            search_results: List[SearchResult],
            filters: SearchFilter
    ) -> List[SearchResult]:
        """应用搜索过滤器"""
        filtered_results = []

        for result in search_results:
            document = result.document

            # 文件名过滤
            if filters.file_names:
                if not any(name.lower() in document.file_name.lower() for name in filters.file_names):
                    continue

            # 文件类型过滤
            if filters.file_types:
                file_ext = document.file_name.split('.')[-1].lower()
                if file_ext not in [ft.lower() for ft in filters.file_types]:
                    continue

            # 日期范围过滤
            if filters.date_range:
                doc_date = datetime.fromisoformat(document.created_at.replace('Z', '+00:00'))
                start_date = datetime.fromisoformat(filters.date_range[0])
                end_date = datetime.fromisoformat(filters.date_range[1])
                if not (start_date <= doc_date <= end_date):
                    continue

            # 元数据过滤
            if filters.metadata_filters:
                match = True
                for key, value in filters.metadata_filters.items():
                    if key not in document.metadata or document.metadata[key] != value:
                        match = False
                        break
                if not match:
                    continue

            filtered_results.append(result)

        return filtered_results

    async def _convert_to_enhanced_results(
            self,
            search_results: List[SearchResult]
    ) -> List[EnhancedSearchResult]:
        """转换为增强搜索结果"""
        enhanced_results = []

        for result in search_results:
            document = result.document

            chunk_info = {
                'chunk_index': document.chunk_index,
                'text': document.text,
                'score': result.score,
                'text_length': len(document.text)
            }

            enhanced_result = EnhancedSearchResult(
                doc_id=document.doc_id,
                file_name=document.file_name,
                file_path=document.file_path,
                chunks=[chunk_info],
                max_score=result.score,
                avg_score=result.score,
                rank=result.rank,
                metadata=document.metadata
            )

            enhanced_results.append(enhanced_result)

        return enhanced_results

    async def _merge_chunks(
            self,
            results: List[EnhancedSearchResult]
    ) -> List[EnhancedSearchResult]:
        """合并同一文档的多个块"""
        doc_groups = defaultdict(list)

        # 按文档ID分组
        for result in results:
            doc_groups[result.doc_id].append(result)

        merged_results = []

        for doc_id, doc_results in doc_groups.items():
            if len(doc_results) == 1:
                merged_results.append(doc_results[0])
            else:
                # 合并多个块
                first_result = doc_results[0]
                all_chunks = []
                scores = []

                for result in doc_results:
                    all_chunks.extend(result.chunks)
                    scores.append(result.max_score)

                # 按分数排序块
                all_chunks.sort(key=lambda x: x['score'], reverse=True)

                merged_result = EnhancedSearchResult(
                    doc_id=doc_id,
                    file_name=first_result.file_name,
                    file_path=first_result.file_path,
                    chunks=all_chunks[:5],  # 最多保留5个最佳块
                    max_score=max(scores),
                    avg_score=sum(scores) / len(scores),
                    rank=min(result.rank for result in doc_results),
                    metadata=first_result.metadata
                )

                merged_results.append(merged_result)

        # 重新排序
        merged_results.sort(key=lambda x: x.max_score, reverse=True)
        for i, result in enumerate(merged_results):
            result.rank = i

        return merged_results
