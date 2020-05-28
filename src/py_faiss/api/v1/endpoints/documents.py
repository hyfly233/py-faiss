from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from typing import List, Optional
import logging

from py_faiss.services.document_service import get_document_service

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/upload")
async def upload_document(
        file: UploadFile = File(...),
        user_id: Optional[str] = None
):
    """上传文档"""
    try:
        # 检查文件类型
        document_service = await get_document_service()

        if not document_service.document_processor.is_supported_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {file.filename}"
            )

        # 读取文件内容
        content = await file.read()

        # 处理文档
        result = await document_service.upload_and_process_document(
            file_content=content,
            filename=file.filename,
            user_id=user_id
        )

        return result

    except Exception as e:
        logger.error(f"上传文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{doc_id}/status")
async def get_document_status(doc_id: str):
    """获取文档处理状态"""
    document_service = await get_document_service()
    status = await document_service.get_processing_status(doc_id)

    if status is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    return status