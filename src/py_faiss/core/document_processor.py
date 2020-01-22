
class DocumentProcessor():
    def __init__(self):
        self.embedding_service = None  # Placeholder for the embedding service
        self.chunk_size = 512  # Default chunk size
        self.overlap = 50  # Default overlap size