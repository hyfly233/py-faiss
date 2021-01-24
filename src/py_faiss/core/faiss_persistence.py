import json
import os
import pickle
from datetime import datetime
from typing import Dict, Any

import faiss


class FAISSPersistence:
    """FAISS索引持久化类，负责保存和加载FAISS索引、元数据和配置信息"""
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.index_file = os.path.join(base_path, "faiss_index.bin")
        self.metadata_file = os.path.join(base_path, "metadata.pkl")
        self.config_file = os.path.join(base_path, "config.json")

        os.makedirs(base_path, exist_ok=True)

    def save_index(self, index: faiss.Index, metadata: Dict[str, Any], config: Dict[str, Any]):
        """保存索引、元数据和配置"""
        try:
            # 保存 FAISS 索引
            faiss.write_index(index, self.index_file)

            # 保存元数据（文档信息、向量ID映射等）
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)

            # 保存配置信息（维度、模型名称等）
            config['saved_at'] = datetime.now().isoformat()
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            print(f"索引保存成功: {self.index_file}")

        except Exception as e:
            raise Exception(f"保存失败: {str(e)}")

    def load_index(self) -> tuple:
        """加载索引、元数据和配置"""
        try:
            # 检查文件是否存在
            if not all(os.path.exists(f) for f in [self.index_file, self.metadata_file, self.config_file]):
                return None, None, None

            # 加载 FAISS 索引
            index = faiss.read_index(self.index_file)

            # 加载元数据
            with open(self.metadata_file, 'rb') as f:
                metadata = pickle.load(f)

            # 加载配置
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            print(f"索引加载成功: {index.ntotal} 个向量")
            return index, metadata, config

        except Exception as e:
            raise Exception(f"加载失败: {str(e)}")