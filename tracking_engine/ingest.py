"""
PDF ingestion pipeline with LlamaParse support.
Automatically switches between LlamaParse (accurate) and local parsing (secure).
"""

import uuid
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

from template.config import settings
from template.chunking import chunker
from template.embeddings_store import upsert_chunks

logger = logging.getLogger(__name__)


def ingest_pdf(
    pdf_path: str,
    doc_id: str = None,
    source_name: str = None
) -> Dict[str, Any]:
    """
    Complete ingestion pipeline with LlamaParse support:
    1. Parse PDF (LlamaParse or local)
    2. Chunk intelligently
    3. Embed and index
    
    Returns ingestion statistics.
    """
    start_time = time.time()
    pdf_path = Path(pdf_path)
    
    # Validate file
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    
    # Generate doc_id if not provided
    if doc_id is None:
        doc_id = str(uuid.uuid4())
    
    source_name = source_name or pdf_path.name
    
    logger.info(f"Starting ingestion: {source_name}")
    logger.info(f" Doc ID: {doc_id}")
    logger.info(f" Size: {file_size_mb:.2f}MB")
    logger.info(f" Parser: {'LlamaParse' if settings.USE_LLAMAPARSE else 'Local'}")
    
    try:
        # --- Step 1: Parse PDF ---
        logger.info("Step 1/4: Parsing PDF...")
        
        if settings.USE_LLAMAPARSE:
            # Use LlamaParse for accuracy
            from template.llamaparse_parser import create_llamaparse_parser
            parser = create_llamaparse_parser(settings.LLAMAPARSE_API_KEY)
            elements = parser.parse(str(pdf_path))
        else:
            # Use local parser for security
            from template.pdf_parser import pdf_parser
            elements = pdf_parser.parse(str(pdf_path))
        
        if not elements:
            raise ValueError("No content extracted from PDF")
        
        logger.info(f"Extracted {len(elements)} elements!")
        
        # --- Step 2: Merge tables with context (optional enhancement) ---
        logger.info("Step 2/4: Merging tables with context.")
        elements = _merge_tables_with_context(elements)
        logger.info(f"Enhanced {len(elements)} elements")
        
        # --- Step 3: Chunk elements ---
        logger.info("Step 3/4: Chunking...")
        chunks = chunker.chunk_elements(elements, doc_id)
        
        # Add source metadata to all chunks
        for chunk in chunks:
            chunk["meta"]["source_name"] = source_name
            chunk["meta"]["doc_id"] = doc_id
            chunk["meta"]["parser_used"] = "LlamaParse" if settings.USE_LLAMAPARSE else "Local"
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # --- Step 4: Embed and index ---
        logger.info("Step 4/4: Embedding and indexing.")
        num_indexed = upsert_chunks(chunks)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Ingestion complete in {processing_time:.2f}s!")
        
        # Detailed statistics
        element_breakdown = _count_element_types(elements)
        chunk_breakdown = _count_chunk_types(chunks)
        
        # Calculate average chunk size
        avg_chunk_size = sum(len(c['text']) for c in chunks) / len(chunks) if chunks else 0
        
        return {
            "status": "success",
            "doc_id": doc_id,
            "source_name": source_name,
            "parser_used": "LlamaParse" if settings.USE_LLAMAPARSE else "Local",
            "processing_time_seconds": round(processing_time, 2),
            "processing_time_ms": round(processing_time * 1000, 2),
            "file_size_mb": round(file_size_mb, 2),
            "num_chunks": len(chunks),
            "stats": {
                "elements_extracted": len(elements),
                "element_breakdown": element_breakdown,
                "chunks_created": len(chunks),
                "chunk_breakdown": chunk_breakdown,
                "chunks_indexed": num_indexed,
                "avg_chunk_size_chars": round(avg_chunk_size, 0),
            }
        }
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


def _merge_tables_with_context(elements: List[Any]) -> List[Any]:
    """
    Merge tables with nearby context (fund names, section headers).
    This helps preserve relationships between headers and table data.
    """
    merged = []
    
    for i, el in enumerate(elements):
        element_type = getattr(el, "element_type", "")
        
        if element_type == "table":
            context_before = ""
            context_after = ""
            
            # Get text just before table (often fund name or section title)
            if i > 0:
                prev_elem = elements[i-1]
                prev_type = getattr(prev_elem, "element_type", "")
                if prev_type in ["text", "header"]:
                    context_before = getattr(prev_elem, "text", "").strip()
            
            # Get text just after table (less common but useful)
            if i + 1 < len(elements):
                next_elem = elements[i+1]
                next_type = getattr(next_elem, "element_type", "")
                if next_type in ["text", "header"]:
                    context_after = getattr(next_elem, "text", "").strip()
            
            # Combine context with table
            table_text = getattr(el, "text", "")
            
            # Add context as prefix if available
            if context_before:
                # Check if context is a fund name or header
                if len(context_before) < 200 and ("Fund" in context_before or "fund" in context_before):
                    table_text = f"FUND: {context_before}\n\n{table_text}"
            
            el.text = table_text
            
            # Update metadata
            if not hasattr(el, 'metadata') or el.metadata is None:
                el.metadata = {}
            el.metadata['has_context'] = bool(context_before or context_after)
            if context_before:
                el.metadata['context_before'] = context_before[:100]  # Store first 100 chars
        
        merged.append(el)
    
    return merged


def _count_element_types(elements: List) -> Dict[str, int]:
    """Count elements by type"""
    from collections import Counter
    return dict(Counter(getattr(e, 'element_type', 'unknown') for e in elements))


def _count_chunk_types(chunks: List[Dict]) -> Dict[str, int]:
    """Count chunks by element type"""
    from collections import Counter
    return dict(Counter(c['meta'].get('element_type', 'unknown') for c in chunks))


def delete_document(doc_id: str) -> Dict[str, Any]:
    """
    Delete document and all its chunks from Qdrant.
    
    Args:
        doc_id: Document ID to delete
        
    Returns:
        Deletion result
    """
    logger.info(f"Deleting document: {doc_id}.")
    
    from template.embeddings_store import embeddings_store
    
    try:
        success = embeddings_store.delete_by_doc_id(doc_id)
        
        if success:
            logger.info(f"Document deleted successfully")
            return {
                "status": "success",
                "doc_id": doc_id,
                "message": "Document deleted"
            }
        else:
            logger.warning(f"Delete operation completed but may not have found documents")
            return {
                "status": "success",
                "doc_id": doc_id,
                "message": "Delete completed (no documents found or already deleted)"
            }
    
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        return {
            "status": "failed",
            "doc_id": doc_id,
            "error": str(e)
        }


def ingest_directory(directory_path: str) -> List[Dict[str, Any]]:
    """
    Batch ingest all PDFs in a directory.
    
    Args:
        directory_path: Path to directory containing PDFs
        
    Returns:
        List of ingestion results
    """
    directory = Path(directory_path)
    
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    
    pdf_files = list(directory.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in: {directory}")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDFs to ingest")
    
    results = []
    for pdf_file in pdf_files:
        try:
            result = ingest_pdf(str(pdf_file))
            results.append(result)
            logger.info(f"Ingested: {pdf_file.name}")
        except Exception as e:
            logger.error(f"Failed to ingest {pdf_file.name}: {e}")
            results.append({
                "status": "failed",
                "source_name": pdf_file.name,
                "error": str(e)
            })
    
    # Summary
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - successful
    
    logger.info(f"Batch ingestion complete: {successful}/{len(results)} successful")
    if failed > 0:
        logger.warning(f"{failed} files failed.")
    
    return results


def get_collection_stats() -> Dict[str, Any]:
    """
    Get statistics about the current collection.
    
    Returns:
        Collection statistics
    """
    from template.embeddings_store import embeddings_store
    
    info = embeddings_store.get_collection_info()
    
    return {
        "collection_name": info.get('name'),
        "total_vectors": info.get('vectors_count', 0),
        "total_points": info.get('points_count', 0),
        "status": info.get('status'),
        "parser_mode": "LlamaParse" if settings.USE_LLAMAPARSE else "Local",
        "config": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "embed_model": settings.EMBED_MODEL,
            "embed_dim": settings.EMBED_DIM,
        }
    }