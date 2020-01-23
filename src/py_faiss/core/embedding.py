from py_faiss.config import settings

class EmbeddingService():
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.EMBEDDING_MODEL
        self.embedding_dim = settings.EMBEDDING_DIMENSION

    async def initialize(self):
        pass
