import asyncio
import logging
import psutil
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from py_faiss.core.embedding import get_embedding_service
from py_faiss.core.vector_store import get_vector_store
from py_faiss.services.document_service import get_document_service
from py_faiss.services.search_service import get_search_service
from py_faiss.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class HealthStatus(BaseModel):
    """健康状态模型"""
    status: str  # healthy, degraded, unhealthy
    timestamp: str
    version: str
    uptime: float

class ComponentHealth(BaseModel):
    """组件健康状态"""
    name: str
    status: str
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class SystemMetrics(BaseModel):
    """系统指标"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    load_average: List[float]

class DetailedHealthResponse(BaseModel):
    """详细健康检查响应"""
    status: str
    timestamp: str
    uptime: float
    version: str
    components: List[ComponentHealth]
    system_metrics: SystemMetrics
    performance_metrics: Dict[str, Any]


class HealthChecker:
    """健康检查器"""

    def __init__(self):
        self.start_time = time.time()
        self.health_history: List[Dict[str, Any]] = []
        self.max_history = 100  # 保留最近100次检查记录

        # 性能阈值
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 90.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            'response_time_warning': 1.0,
            'response_time_critical': 5.0
        }

    async def check_basic_health(self) -> HealthStatus:
        """基础健康检查"""
        try:
            uptime = time.time() - self.start_time

            # 简单检查：能否正常响应
            status = "healthy"

            return HealthStatus(
                status=status,
                timestamp=datetime.now().isoformat(),
                uptime=uptime
            )

        except Exception as e:
            logger.error(f"基础健康检查失败: {e}")
            return HealthStatus(
                status="unhealthy",
                timestamp=datetime.now().isoformat(),
                uptime=0.0
            )

    async def check_detailed_health(self) -> DetailedHealthResponse:
        """详细健康检查"""
        start_time = time.time()

        try:
            # 检查各个组件
            components = await self._check_all_components()

            # 获取系统指标
            system_metrics = await self._get_system_metrics()

            # 获取性能指标
            performance_metrics = await self._get_performance_metrics()

            # 计算总体状态
            overall_status = self._calculate_overall_status(components, system_metrics)

            # 记录健康检查历史
            health_record = {
                'timestamp': datetime.now().isoformat(),
                'status': overall_status,
                'check_duration': time.time() - start_time,
                'component_count': len(components),
                'healthy_components': len([c for c in components if c.status == 'healthy'])
            }

            self.health_history.append(health_record)
            if len(self.health_history) > self.max_history:
                self.health_history.pop(0)

            return DetailedHealthResponse(
                status=overall_status,
                timestamp=datetime.now().isoformat(),
                uptime=time.time() - self.start_time,
                version="1.0.0",
                components=components,
                system_metrics=system_metrics,
                performance_metrics=performance_metrics
            )

        except Exception as e:
            logger.error(f"详细健康检查失败: {e}")
            raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")

    async def _check_all_components(self) -> List[ComponentHealth]:
        """检查所有组件"""
        components = []

        # 并行检查所有组件
        tasks = [
            self._check_embedding_service(),
            self._check_vector_store(),
            self._check_document_service(),
            self._check_search_service(),
            self._check_storage(),
            self._check_dependencies()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, ComponentHealth):
                components.append(result)
            elif isinstance(result, Exception):
                components.append(ComponentHealth(
                    name="unknown_component",
                    status="unhealthy",
                    error=str(result)
                ))

        return components

    async def _check_embedding_service(self) -> ComponentHealth:
        """检查嵌入服务"""
        start_time = time.time()

        try:
            embedding_service = await get_embedding_service()

            # 测试嵌入生成
            test_text = "健康检查测试文本"
            embedding = await embedding_service.get_embedding(test_text)

            response_time = time.time() - start_time

            # 验证嵌入向量
            if embedding is not None and len(embedding) == settings.EMBEDDING_DIMENSION:
                status = "healthy"
                if response_time > self.thresholds['response_time_warning']:
                    status = "degraded"
            else:
                status = "unhealthy"

            return ComponentHealth(
                name="embedding_service",
                status=status,
                response_time=response_time,
                details={
                    'model_name': embedding_service.model_name,
                    'dimension': len(embedding) if embedding is not None else 0,
                    'device': getattr(embedding_service, 'device', 'unknown')
                }
            )

        except Exception as e:
            return ComponentHealth(
                name="embedding_service",
                status="unhealthy",
                response_time=time.time() - start_time,
                error=str(e)
            )

    async def _check_vector_store(self) -> ComponentHealth:
        """检查向量存储"""
        start_time = time.time()

        try:
            vector_store = await get_vector_store()

            # 获取统计信息
            stats = await vector_store.get_stats()

            response_time = time.time() - start_time

            # 检查索引状态
            index_healthy = (
                    vector_store.index is not None and
                    stats.get('total_chunks', 0) >= 0
            )

            status = "healthy" if index_healthy else "degraded"

            return ComponentHealth(
                name="vector_store",
                status=status,
                response_time=response_time,
                details={
                    'total_documents': stats.get('total_documents', 0),
                    'total_chunks': stats.get('total_chunks', 0),
                    'index_size': stats.get('index_size', 0),
                    'dimension': vector_store.dimension,
                    'index_type': vector_store.index_type
                }
            )

        except Exception as e:
            return ComponentHealth(
                name="vector_store",
                status="unhealthy",
                response_time=time.time() - start_time,
                error=str(e)
            )

    async def _check_document_service(self) -> ComponentHealth:
        """检查文档服务"""
        start_time = time.time()

        try:
            document_service = await get_document_service()

            # 获取统计信息
            stats = await document_service.get_statistics()

            response_time = time.time() - start_time

            # 检查服务状态
            service_healthy = 'error' not in stats

            status = "healthy" if service_healthy else "degraded"

            return ComponentHealth(
                name="document_service",
                status=status,
                response_time=response_time,
                details={
                    'processing_queue': len(document_service.processing_status),
                    'supported_formats': len(document_service.document_processor.get_supported_types()),
                    'temp_dir': str(document_service.document_processor.temp_dir)
                }
            )

        except Exception as e:
            return ComponentHealth(
                name="document_service",
                status="unhealthy",
                response_time=time.time() - start_time,
                error=str(e)
            )

    async def _check_search_service(self) -> ComponentHealth:
        """检查搜索服务"""
        start_time = time.time()

        try:
            search_service = await get_search_service()

            # 获取搜索统计
            stats = await search_service.get_search_statistics()

            response_time = time.time() - start_time

            status = "healthy"

            return ComponentHealth(
                name="search_service",
                status=status,
                response_time=response_time,
                details={
                    'total_searches': stats.get('total_searches', 0),
                    'avg_search_time': stats.get('avg_search_time', 0),
                    'cache_size': stats.get('cache_size', 0),
                    'history_size': stats.get('history_size', 0)
                }
            )

        except Exception as e:
            return ComponentHealth(
                name="search_service",
                status="unhealthy",
                response_time=time.time() - start_time,
                error=str(e)
            )