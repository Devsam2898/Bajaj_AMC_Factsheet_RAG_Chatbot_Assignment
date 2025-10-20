import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ============ LlamaParse Settings ============
    USE_LLAMAPARSE: bool = True  # Toggle between LlamaParse and local parsing
    LLAMAPARSE_API_KEY: str = os.getenv("LLAMAPARSE_API_KEY", "")

    # ============ Qdrant Configuration ============
    QDRANT_HOST: str = os.getenv('QDRANT_HOST', 'localhost')
    QDRANT_PORT: int = int(os.getenv('QDRANT_PORT', 6333))
    QDRANT_URL: str = os.getenv('QDRANT_URL', '')
    QDRANT_API_KEY: str = os.getenv('QDRANT_API_KEY', '')
    QDRANT_COLLECTION: str = os.getenv('QDRANT_COLLECTION', 'factsheets')
    QDRANT_LOCAL_PATH: Path = Path(os.getenv('QDRANT_LOCAL_PATH', './qdrant_local'))
    
    # ============ Embedding Configuration ============
    # FIXED: Use consistent embedding model
    EMBED_MODEL: str = os.getenv('EMBED_MODEL', 'BAAI/bge-small-en-v1.5')
    EMBED_DIM: int = 384  # bge-small-en-v1.5 dimension
    EMBED_BATCH_SIZE: int = 32
    
    # ============ Reranker Configuration ============
    RERANKER_MODEL: str = os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-base')
    RERANKER_BATCH_SIZE: int = 16
    
    # ============ LLM Configuration ============
    LLAMA_MODEL: str = os.getenv('LLAMA_MODEL', 'meta-llama/Llama-3.1-8B-Instruct')
    LLAMA_DEVICE: str = os.getenv('LLAMA_DEVICE', 'cuda')
    
    # Quantization settings (4-bit for efficiency)
    USE_4BIT_QUANT: bool = os.getenv('USE_4BIT_QUANT', 'true').lower() == 'true'
    BNB_4BIT_COMPUTE_DTYPE: str = 'float16'
    BNB_4BIT_QUANT_TYPE: str = 'nf4'
    
    # Generation parameters
    MAX_NEW_TOKENS: int = int(os.getenv('MAX_NEW_TOKENS', 512))
    TEMPERATURE: float = float(os.getenv('TEMPERATURE', 0.3))
    TOP_P: float = float(os.getenv('TOP_P', 0.9))
    
    # ============ Chunking Configuration ============
    CHUNK_SIZE: int = int(os.getenv('CHUNK_SIZE', 512))
    CHUNK_OVERLAP: int = int(os.getenv('CHUNK_OVERLAP', 50))
    
    # ============ Retrieval Configuration ============
    TOP_K: int = int(os.getenv('TOP_K', 50))  # Increased for better reranking
    RERANK_TOP_N: int = int(os.getenv('RERANK_TOP_N', 15))  # Final context size
    
    # ============ Redis Configuration ============
    REDIS_HOST: str = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT: int = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB: int = int(os.getenv('REDIS_DB', 0))
    REDIS_PASSWORD: str = os.getenv('REDIS_PASSWORD', '')
    CACHE_TTL: int = int(os.getenv('CACHE_TTL', 3600))  # 1 hour
    
    # ============ Conversation Memory ============
    MAX_HISTORY_TURNS: int = int(os.getenv('MAX_HISTORY_TURNS', 5))
    MEMORY_WINDOW_TOKENS: int = int(os.getenv('MEMORY_WINDOW_TOKENS', 2048))
    
    # ============ PDF Processing ============
    TEMP_DIR: Path = Path(os.getenv('TEMP_DIR', '/tmp/factsheets'))
    MAX_PDF_SIZE_MB: int = int(os.getenv('MAX_PDF_SIZE_MB', 50))
    
    # OCR settings (disabled by default for Modal compatibility)
    USE_OCR: bool = os.getenv('USE_OCR', 'false').lower() == 'true'
    OCR_LANG: str = os.getenv('OCR_LANG', 'en')
    
    # ============ API Configuration ============
    API_TITLE: str = "AMC Factsheet RAG API"
    API_VERSION: str = "2.0.0"
    DEBUG: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    
    # ============ Hugging Face ============
    HF_TOKEN: str = os.getenv('HF_TOKEN', '')
    
    class Config:
        env_file = '.env'
        case_sensitive = True


# Singleton instance
settings = Settings()

# Ensure temp directory exists
settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)