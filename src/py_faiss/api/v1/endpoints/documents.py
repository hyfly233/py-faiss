from fastapi import APIRouter

documents_router = APIRouter()


@documents_router.get("/test")
async def test():
    return [{"test": "documents"}]


def router() -> APIRouter:
    return documents_router
