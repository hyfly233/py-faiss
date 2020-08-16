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

