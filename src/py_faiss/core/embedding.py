import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List

import aiohttp
import numpy as np

from py_faiss.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, base_url: str = None, model_name: str = None, dimension: int = None, timeout: int = 30,
                 max_retries: int = 3):
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.timeout = timeout
        self.max_retries = max_retries

        self.embeddings_url = f"{self.base_url}/api/embeddings"
        self.tags_url = f"{self.base_url}/api/tags"

        # 连接池配置
        self.session = None
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def initialize(self):
        try:
            # 创建 aiohttp 会话
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )

            # 检查 Ollama 连接
            await self._check_ollama_connection()

            # 验证模型可用性
            await self._verify_model()

            # 测试 embedding 生成
            await self._test_embedding()

            logger.info(f"Embedding service initialized with model: {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            if self.session:
                await self.session.close()
                self.session = None
            raise

    async def _check_ollama_connection(self):
        """测试与 Ollama 的连接"""
        try:
            async with self.session.get(self.tags_url) as response:
                if response.status == 200:
                    logger.info("Ollama connection successful")
                else:
                    raise Exception(f"Ollama returned status {response.status}")
        except Exception as e:
            raise Exception(f"Cannot connect to Ollama at {self.base_url}: {e}")

    async def _verify_model(self):
        """验证模型是否可用"""
        try:
            async with self.session.get(self.tags_url) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    model_names = [model['name'] for model in models]

                    # 检查模型是否存在（支持 model:latest 格式）
                    if self.model_name in model_names or f"{self.model_name}:latest" in model_names:
                        logger.info(f"Model {self.model_name} is available")
                    else:
                        available_models = ", ".join(model_names)
                        raise Exception(
                            f"Model {self.model_name} not found. "
                            f"Available models: {available_models}"
                        )
                else:
                    raise Exception("Cannot retrieve model list")
        except Exception as e:
            raise Exception(f"Model verification failed: {e}")

    async def _test_embedding(self):
        """测试 embedding 生成"""
        try:
            test_text = "测试文本"
            embedding = await self.get_embedding(test_text)

            if len(embedding) != self.dimension:
                logger.warning(
                    f"Expected dimension {self.dimension}, got {len(embedding)}"
                )
                # 更新实际维度
                self.dimension = len(embedding)

            logger.info(f"Embedding test successful, dimension: {self.dimension}")

        except Exception as e:
            raise Exception(f"Embedding test failed: {e}")

    async def get_embedding(self, text: str) -> np.ndarray:
        """
        获取单个文本的 embedding

        Args:
            text: 输入文本

        Returns:
            embedding 向量
        """
        if not self.session:
            raise Exception("Embedding service not initialized")

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        for attempt in range(self.max_retries):
            try:
                pass
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) * 0.5  # 指数退避
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"Failed to get embedding after {self.max_retries} attempts: {e}")



    async def get_embeddings_batch(self, texts: List[str], batch_size: int = 10,
                                   show_progress: bool = True) -> np.ndarray:
        pass

    async def _process_batch_concurrent(self, texts: List[str]) -> List[np.ndarray]:
        pass
