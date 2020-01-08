from fastapi import APIRouter

search_router = APIRouter()


@search_router.get("/test")
async def test():
    return [{"test": "search"}]


def router() -> APIRouter:
    return search_router
