"""
Modal Labs deployment configuration for L40S GPU.
Optimized for production use with LlamaParse support.
"""

import modal
import os

# ============ Get absolute paths (only used during local build) ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Check if we're being run from tracking_engine or from Bajaj
if os.path.basename(SCRIPT_DIR) == "tracking_engine":
    BAJAJ_DIR = os.path.dirname(SCRIPT_DIR)
else:
    BAJAJ_DIR = SCRIPT_DIR

TEMPLATE_DIR = os.path.join(BAJAJ_DIR, "template")
OPERATIONS_DIR = os.path.join(BAJAJ_DIR, "operations")
TRACKING_DIR = os.path.join(BAJAJ_DIR, "tracking_engine")

# ============ Modal Image Configuration ============
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        # System dependencies
        "libgomp1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgl1-mesa-glx",
        "ghostscript",
        "default-jre",  # Required for Tabula/Camelot
        "git",
    )
    # ------------------------------------------------------------------
    # 1. Core Numerical/ML Stack (Compatible versions installed together)
    #    - PyTorch 2.3.1 supports NumPy 2.0.0+ natively.
    # ------------------------------------------------------------------
    .pip_install(
        # Core Utilities and Numerical Libraries
        "pip>=24.2",
        "setuptools>=69.0.0",
        "wheel",
        "numpy==2.0.0",           # NumPy 2.0.0+
        "scipy==1.13.1",
        "scikit-learn==1.5.1",
    )
    # Use run_commands to install the main libraries to enforce the index-url
    # for PyTorch, which is a common practice for clean builds.
    .run_commands(
        # PyTorch 2.3.1 (compatible with NumPy 2.0.0)
        "pip install --upgrade torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        # Hugging Face ML Stack
        "transformers==4.44.2",
        "accelerate==0.26.1",
        "bitsandbytes==0.43.1",   # UPDATED for 2.3.1 compatibility
        "sentence-transformers==2.3.1",
    )
    # ------------------------------------------------------------------
    # 2. PDF Processing and Camelot
    # ------------------------------------------------------------------
    .pip_install(
        "PyMuPDF==1.23.21",
        "opencv-python-headless==4.9.0.80",
        "tabula-py==2.9.0",
        "camelot-py[base]==0.11.0",
        "pandas==2.1.4",  # Explicit version for stability
    )
    # ------------------------------------------------------------------
    # 3. LlamaParse - NEW ADDITION
    # ------------------------------------------------------------------
    .pip_install(
        "llama-parse==0.5.11",  # Updated for Pydantic 2.x compatibility
        "llama-index-core==0.12.4",  # Updated for compatibility
        "nest-asyncio==1.6.0",  # Required for LlamaParse in FastAPI
    )
    # ------------------------------------------------------------------
    # 4. Remaining Packages
    # ------------------------------------------------------------------
    .pip_install(
        "fastapi==0.109.0",
        "uvicorn[standard]==0.27.0",
        "python-multipart==0.0.6",
        "pydantic==2.9.2",  # Updated for LlamaParse compatibility
        "pydantic-settings==2.6.1",  # Updated to match
        "qdrant-client==1.7.3",
        "redis==5.0.1",
        "hiredis==2.3.2",
        "python-dotenv==1.0.0",
        "tqdm==4.66.1",
        "requests==2.31.0",
    )
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
    })
)

# ============ Modal App ============
app = modal.App("amc-factsheet-rag", image=image)

# ============ Copy folders one by one ============
image_with_template = image.add_local_dir(
    TEMPLATE_DIR,
    remote_path="/root/bajaj/template"
)

image_with_operations = image_with_template.add_local_dir(
    OPERATIONS_DIR,
    remote_path="/root/bajaj/operations"
)

image_with_tracking = image_with_operations.add_local_dir(
    TRACKING_DIR,
    remote_path="/root/bajaj/tracking_engine"
)

# ============ Volume for Qdrant (persistent storage) ============
qdrant_volume = modal.Volume.from_name(
    "qdrant-data",
    create_if_missing=True
)

# ============ GPU Configuration ============
GPU_CONFIG = "L40S"

# ============ Secrets ============
SECRETS = [
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("llamaparse-secret"),  # NEW: LlamaParse API key
]

# ============ FastAPI Deployment ============
@app.function(
    image=image_with_tracking,
    gpu=GPU_CONFIG,
    secrets=SECRETS,
    volumes={"/data/qdrant": qdrant_volume},
    timeout=600,
    scaledown_window=300,
    memory=16384,
    max_containers=10,
)
@modal.asgi_app()
def fastapi_app():
    """
    Deploy FastAPI app with all components loaded on startup.
    """
    import os
    import sys
    
    # Add ALL necessary folders to Python path
    sys.path.insert(0, "/root/bajaj")
    sys.path.insert(0, "/root/bajaj/template")
    sys.path.insert(0, "/root/bajaj/operations")
    sys.path.insert(0, "/root/bajaj/tracking_engine")
    
    # Set environment variables
    os.environ["QDRANT_LOCAL_PATH"] = "/data/qdrant"
    os.environ["USE_4BIT_QUANT"] = "true"
    os.environ["USE_LLAMAPARSE"] = "true"  # NEW: Enable LlamaParse
    
    # Import from tracking_engine
    from tracking_engine.main import app
    return app


# ============ CLI for testing ============
@app.local_entrypoint()
def test_deployment():
    """
    Test the deployment locally.
    Usage: modal run modal_deploy.py
    """
    print("Testing Modal deployment...")
    
    # Test the actual endpoint
    import requests
    base_url = "https://devsam2898--amc-factsheet-rag-fastapi-app.modal.run"
    
    print(f"\nTesting endpoint: {base_url}")
    
    try:
        print("Sending request (may take 2-5 min on first call)...")
        response = requests.get(f"{base_url}/", timeout=300)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except requests.exceptions.Timeout:
        print("Request timed out. Check Modal logs:")
        print(" modal app logs amc-factsheet-rag")
    except Exception as e:
        print(f"Error: {e}")
        print("\n💡 Check Modal dashboard:")
        print("   https://modal.com/apps/devsam2898/main/deployed/amc-factsheet-rag")


# ============ Test LlamaParse Function ============
@app.function(
    image=image_with_tracking,
    secrets=SECRETS,
    timeout=300,
)
def test_llamaparse(pdf_url: str = None):
    """
    Test LlamaParse parsing on Modal.
    Usage:
        modal run modal_deploy.py::test_llamaparse
        modal run modal_deploy.py::test_llamaparse --pdf-url="https://example.com/test.pdf"
    """
    import sys
    import os
    
    sys.path.insert(0, "/root/bajaj")
    sys.path.insert(0, "/root/bajaj/template")
    
    os.environ["USE_LLAMAPARSE"] = "true"
    
    from template.config import settings
    from template.llamaparse_parser import create_llamaparse_parser
    import requests
    from pathlib import Path
    
    # Use default test PDF if not provided
    if not pdf_url:
        pdf_url = "https://www.bajajamc.com/downloads/factsheet/bajaj_finserv_liquid_fund.pdf"
    
    print(f"Downloading: {pdf_url}")
    response = requests.get(pdf_url, timeout=60)
    response.raise_for_status()
    
    temp_path = Path("/tmp/test.pdf")
    with open(temp_path, "wb") as f:
        f.write(response.content)
    
    print(f"Downloaded to: {temp_path}")
    
    # Parse with LlamaParse
    print(f"Parsing with LlamaParse (this may take 30-60 seconds)...")
    parser = create_llamaparse_parser(settings.LLAMAPARSE_API_KEY)
    elements = parser.parse(str(temp_path))
    
    print(f"\nParsing complete!")
    print(f" Total elements: {len(elements)}")
    
    # Analyze results
    tables = [e for e in elements if e.element_type == 'table']
    print(f"   Tables found: {len(tables)}")
    
    if tables:
        print(f"\nFirst table preview:")
        print(f" Page: {tables[0].page_number}")
        print(f" Length: {len(tables[0].text)} chars")
        print(f" Metadata: {tables[0].metadata}")
        print(f"\n Content (first 500 chars):")
        print(tables[0].text[:500])
    
    temp_path.unlink()
    return {
        "status": "success", 
        "elements": len(elements), 
        "tables": len(tables),
        "parser": "LlamaParse"
    }


# ============ Batch Ingestion Function ============
@app.function(
    image=image_with_tracking,
    gpu=GPU_CONFIG,
    secrets=SECRETS,
    volumes={"/data/qdrant": qdrant_volume},
    timeout=3600,
)
def batch_ingest_pdfs(pdf_urls: list[str]):
    """
    Batch ingest PDFs from URLs.
    Usage:
        modal run modal_deploy.py::batch_ingest_pdfs --pdf-urls='["url1", "url2"]'
    """
    import os
    import sys
    sys.path.insert(0, "/root/bajaj")
    sys.path.insert(0, "/root/bajaj/template")
    sys.path.insert(0, "/root/bajaj/operations")
    sys.path.insert(0, "/root/bajaj/tracking_engine")
    
    os.environ["QDRANT_LOCAL_PATH"] = "/data/qdrant"
    os.environ["USE_LLAMAPARSE"] = "true"
    
    from tracking_engine.ingest import ingest_pdf
    import requests
    from pathlib import Path
    
    results = []
    for url in pdf_urls:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            filename = url.split("/")[-1]
            temp_path = Path(f"/tmp/{filename}")
            
            with open(temp_path, "wb") as f:
                f.write(response.content)
            
            result = ingest_pdf(str(temp_path), source_name=filename)
            results.append(result)
            print(f"✅ Ingested: {filename}")
            temp_path.unlink()
        except Exception as e:
            print(f"Failed: {e}")
            results.append({"status": "failed", "url": url, "error": str(e)})
    
    return results


# ============ Model Warmup Function ============
@app.function(
    image=image_with_tracking,
    gpu=GPU_CONFIG,
    secrets=SECRETS,
)
def warmup_models():
    """
    Pre-load models to cache them for faster cold starts.
    Run this after deployment to warm up the container.
    """
    import sys
    sys.path.insert(0, "/root/bajaj")
    sys.path.insert(0, "/root/bajaj/template")
    sys.path.insert(0, "/root/bajaj/operations")
    sys.path.insert(0, "/root/bajaj/tracking_engine")
    
    print("Warming up models...")
    from tracking_engine.query_engine import query_engine
    from template.embeddings_store import embeddings_store
    
    test_result = query_engine.retrieve_and_answer(
        "What is the expense ratio?",
        conversation_id=None,
        include_history=False
    )
    print("Models warmed up!")
    return {"status": "warm", "test_query_successful": True}