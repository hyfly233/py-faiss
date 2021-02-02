## 简介

通过 ollama 运行的 embedding 模型，如 bge-m3，对文件中的文本进行向量化处理。再将向量化后的数据存储到向量数据库 faiss 中。使用
http API 进行数据查询。

## 架构说明

![架构图](./img/架构图.png)