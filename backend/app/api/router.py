from fastapi import APIRouter

from app.api.routes import conversations, documents, health, memories, playground

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(conversations.router, prefix="/conversations", tags=["conversations"])
api_router.include_router(memories.router, prefix="/memories", tags=["memories"])
api_router.include_router(playground.router, prefix="/playground", tags=["playground"])
