from fastapi import APIRouter

health_router = APIRouter()


@health_router.get("/test")
async def test():
    return [{"test": "health"}]


def router() -> APIRouter:
    return health_router
