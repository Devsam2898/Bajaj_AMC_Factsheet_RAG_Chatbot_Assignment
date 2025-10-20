"""
Production FastAPI application with error handling and monitoring.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import uuid
import time
from pathlib import Path

from template.config import settings
from tracking_engine.ingest import ingest_pdf
from tracking_engine.query_engine import query_engine
from template.cache import get_cached, set_cached
from template.embeddings_store import embeddings_store
from operations.conversation_memory import conversation_manager

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============ FastAPI App ============
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Secure RAG API for AMC factsheet analysis"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Request/Response Models ============
class QueryRequest(BaseModel):
    query: str = Field(..., description="User question")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for multi-turn")
    use_cache: bool = Field(True, description="Enable caching")
    include_history: bool = Field(True, description="Include conversation history")


class QueryResponse(BaseModel):
    answer: str
    citations: list
    num_sources: int
    cached: bool
    processing_time_ms: float
    conversation_id: Optional[str] = None


class IngestResponse(BaseModel):
    status: str
    doc_id: str
    source_name: str
    file_size_mb: float
    elements_extracted: int
    chunks_created: int
    chunks_indexed: int
    element_types: dict
    processing_time_ms: float


# ============ Health Check ============
@app.get("/")
async def root():
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        # Check Qdrant
        collection_info = embeddings_store.get_collection_info()
        
        return {
            "status": "healthy",
            "qdrant": {
                "connected": True,
                "collection": settings.QDRANT_COLLECTION,
                "points": collection_info.get("points_count", 0)
            },
            "models": {
                "embedding": settings.EMBED_MODEL,
                "reranker": settings.RERANKER_MODEL,
                "llm": settings.LLAMA_MODEL
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")


# ============ Ingestion Endpoint ============
@app.post("/ingest")  # Remove response_model for now
async def ingest_endpoint(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Query(None, description="Optional custom document ID")
):
    """
    Ingest a PDF factsheet.
    """
    start_time = time.time()
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    try:
        # Save temporarily
        temp_path = settings.TEMP_DIR / f"{uuid.uuid4()}_{file.filename}"
        
        with open(temp_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Received file: {file.filename} ({len(content)} bytes)")
        
        # Ingest
        result = ingest_pdf(
            pdf_path=str(temp_path),
            doc_id=doc_id,
            source_name=file.filename
        )
        
        # Cleanup
        temp_path.unlink()
        
        # Return complete result (let FastAPI handle serialization)
        return result
    
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        
        # Cleanup on error
        if temp_path.exists():
            temp_path.unlink()
        
        raise HTTPException(status_code=500, detail=str(e))

# ============ Query Endpoint ============
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query the RAG system.
    - Supports conversation history
    - Caching for repeated queries
    - Citation tracking
    """
    start_time = time.time()
    
    # Generate conversation ID if not provided
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    # Check cache
    if request.use_cache:
        cached = get_cached(request.query, conversation_id)
        if cached:
            cached["cached"] = True
            cached["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
            cached["conversation_id"] = conversation_id
            logger.info(f"Cache hit for: {request.query[:50]}...")
            return cached
    
    try:
        # Query RAG system
        result = query_engine.retrieve_and_answer(
            query=request.query,
            conversation_id=conversation_id if request.include_history else None,
            include_history=request.include_history
        )
        
        # Cache result
        if request.use_cache:
            set_cached(request.query, result, conversation_id)
        
        # Add timing and conversation ID
        result["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
        result["conversation_id"] = conversation_id
        
        logger.info(
            f"Query processed in {result['processing_time_ms']}ms"
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Conversation Management ============
@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and its history"""
    try:
        conversation_manager.delete_conversation(conversation_id)
        return {"status": "deleted", "conversation_id": conversation_id}
    except Exception as e:
        logger.error(f"Delete conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{conversation_id}/history")
async def get_conversation_history(conversation_id: str):
    """Get conversation history"""
    try:
        conversation = conversation_manager.get_conversation(conversation_id)
        history = [
            {
                "query": turn.query,
                "response": turn.response,
                "timestamp": turn.timestamp
            }
            for turn in conversation.history
        ]
        return {"conversation_id": conversation_id, "history": history}
    except Exception as e:
        logger.error(f"Get history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Document Management ============
@app.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all its chunks from the vector store"""
    try:
        success = embeddings_store.delete_by_doc_id(doc_id)
        if success:
            return {"status": "deleted", "doc_id": doc_id}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collection/stats")
async def collection_stats():
    """Get vector store statistics"""
    try:
        stats = embeddings_store.get_collection_info()
        return stats
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/collection/clear")
async def clear_collection():
    """Delete all data and recreate collection"""
    try:
        from template.embeddings_store import embeddings_store
        
        # Delete collection
        embeddings_store.qdrant.delete_collection(settings.QDRANT_COLLECTION)
        
        # Recreate
        embeddings_store._ensure_collection()
        
        return {
            "status": "success",
            "message": "Collection cleared and recreated"
        }
    except Exception as e:
        logger.error(f"Clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Error Handlers ============
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# ============ Startup/Shutdown ============
@app.on_event("startup")
async def startup_event():
    logger.info("Starting AMC Factsheet RAG API")
    logger.info(f" Environment: {'DEBUG' if settings.DEBUG else 'PRODUCTION'}")
    logger.info(f" Collection: {settings.QDRANT_COLLECTION}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down AMC Factsheet RAG API")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )