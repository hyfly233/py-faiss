import faiss
import pickle
import json
import os
from typing import Dict, List, Any
from datetime import datetime

class FAISSPersistence:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.index_file = os.path.join(base_path, "faiss_index.bin")
        self.metadata_file = os.path.join(base_path, "metadata.pkl")
        self.config_file = os.path.join(base_path, "config.json")

        os.makedirs(base_path, exist_ok=True)