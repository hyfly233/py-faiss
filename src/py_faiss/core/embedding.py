from concurrent.futures import ThreadPoolExecutor

from py_faiss.config import settings


class EmbeddingService():
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
        pass
