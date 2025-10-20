"""
Enhanced vector store with hybrid search capabilities.
Combines semantic search with metadata filtering for precision.
"""

import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams, Distance, PointStruct, 
    Filter, FieldCondition, MatchValue, MatchAny
)
from template.config import settings
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class EnhancedEmbeddingsStore:
    """
    Advanced embeddings store with hybrid search and metadata filtering.
    """
    
    def __init__(self):
        self.model = None
        self.qdrant = None
        self._init_embedding_model()
        self._init_qdrant()
    
    def _init_embedding_model(self):
        """Initialize embedding model"""
        logger.info(f"Loading embedding model: {settings.EMBED_MODEL}")
        self.model = SentenceTransformer(settings.EMBED_MODEL)
        
        actual_dim = self.model.get_sentence_embedding_dimension()
        if actual_dim != settings.EMBED_DIM:
            logger.warning(f"Updating EMBED_DIM: {settings.EMBED_DIM} → {actual_dim}")
            settings.EMBED_DIM = actual_dim
        
        logger.info(f"Embedding model loaded (dim={actual_dim})")
    
    def _init_qdrant(self):
        """Initialize Qdrant client"""
        if settings.QDRANT_URL:
            self.qdrant = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY
            )
            logger.info(f"Qdrant cloud connected")
        else:
            self.qdrant = QdrantClient(path=str(settings.QDRANT_LOCAL_PATH))
            logger.info(f"Qdrant embedded: {settings.QDRANT_LOCAL_PATH}")
        
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection with optimized settings"""
        try:
            collections = self.qdrant.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if settings.QDRANT_COLLECTION not in collection_names:
                logger.info(f"Creating collection: {settings.QDRANT_COLLECTION}")
                self.qdrant.create_collection(
                    collection_name=settings.QDRANT_COLLECTION,
                    vectors_config=VectorParams(
                        size=settings.EMBED_DIM,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Collection created: {settings.QDRANT_COLLECTION}")
            else:
                logger.info(f"Collection exists: {settings.QDRANT_COLLECTION}")
        
        except Exception as e:
            logger.error(f"Collection creation failed: {e}")
            raise
    
    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Embed and upsert chunks with rich metadata.
        """
        if not chunks:
            return 0
        
        try:
            texts = [c["text"] for c in chunks]
            metadatas = [c.get("meta", {}) for c in chunks]
            
            logger.info(f"Embedding {len(texts)} chunks...")
            
            # Batch encode
            embeddings = self.model.encode(
                texts,
                batch_size=settings.EMBED_BATCH_SIZE,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            
            # Create points with enriched metadata
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[i].tolist(),
                    payload={
                        "text": texts[i],
                        **metadatas[i]
                    }
                )
                for i in range(len(chunks))
            ]
            
            # Upsert to Qdrant
            self.qdrant.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=points,
                wait=True
            )
            
            logger.info(f"Indexed {len(chunks)} chunks")
            return len(chunks)
        
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            raise
    
    def search(
        self, 
        query: str, 
        top_k: int = None,
        filter_dict: Dict[str, Any] = None,
        boost_tables: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search with semantic + metadata filtering.
        
        Args:
            query: Search query
            top_k: Number of results
            filter_dict: Metadata filters
            boost_tables: Prioritize table results for financial queries
        """
        top_k = top_k or settings.TOP_K
        
        try:
            # Encode query
            query_vector = self.model.encode(
                [query],
                normalize_embeddings=True
            )[0].tolist()
            
            # Build filter
            query_filter = self._build_filter(query, filter_dict, boost_tables)
            
            # Search Qdrant
            hits = self.qdrant.search(
                collection_name=settings.QDRANT_COLLECTION,
                query_vector=query_vector,
                limit=top_k * 2,  # Get more results for reranking
                query_filter=query_filter
            )
            
            # Rerank results based on query type
            reranked_hits = self._rerank_results(hits, query, top_k)
            
            # Format results
            results = []
            for hit in reranked_hits:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "metadata": {
                        k: v for k, v in hit.payload.items() 
                        if k != "text"
                    }
                })
            
            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            return results
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _build_filter(
        self, 
        query: str, 
        filter_dict: Optional[Dict] = None,
        boost_tables: bool = True
    ) -> Optional[Filter]:
        """
        Build intelligent filter based on query type.
        """
        conditions = []
        
        # Apply user-provided filters
        if filter_dict:
            for key, value in filter_dict.items():
                if isinstance(value, list):
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchAny(any=value)
                        )
                    )
                else:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
        
        # Smart filtering based on query keywords
        query_lower = query.lower()
        
        # Financial metrics queries → prioritize tables
        if boost_tables and any(term in query_lower for term in [
            'nav', 'return', 'allocation', 'holding', 'expense ratio', 
            'performance', '%', 'percentage', 'fund manager', 'ceo'
        ]):
            # Don't filter out tables, but we'll boost them in reranking
            pass
        
        if conditions:
            return Filter(must=conditions)
        
        return None
    
    def _rerank_results(self, hits, query: str, top_k: int):
        """
        Rerank results based on query type and metadata.
        """
        query_lower = query.lower()
        
        # Detect query type
        is_numerical_query = any(term in query_lower for term in [
            '%', 'percent', 'return', 'ratio', 'nav', 'allocation'
        ])
        is_personnel_query = any(term in query_lower for term in [
            'manager', 'ceo', 'team', 'who'
        ])
        
        scored_hits = []
        for hit in hits:
            boost_score = hit.score
            element_type = hit.payload.get('element_type', '')
            metadata = hit.payload
            
            # Boost tables for numerical queries
            if is_numerical_query and element_type == 'table':
                if metadata.get('contains_numbers') or metadata.get('contains_percentages'):
                    boost_score *= 1.3
            
            # Boost personnel tables for people queries
            if is_personnel_query and element_type == 'table':
                table_types = metadata.get('table_type', [])
                if 'personnel' in table_types:
                    boost_score *= 1.5
            
            # Boost if searchable keywords match
            searchable = metadata.get('searchable_keywords', '')
            if searchable:
                # Count keyword matches
                matches = sum(1 for keyword in query_lower.split() if keyword in searchable.lower())
                if matches > 0:
                    boost_score *= (1 + 0.1 * matches)
            
            scored_hits.append((boost_score, hit))
        
        # Sort by boosted score
        scored_hits.sort(key=lambda x: x[0], reverse=True)
        
        return [hit for _, hit in scored_hits[:top_k]]
    
    def search_with_context(
        self,
        query: str,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search and include surrounding context for better LLM understanding.
        """
        results = self.search(query, top_k)
        
        # Enrich results with context
        for result in results:
            metadata = result['metadata']
            context_parts = []
            
            # Add preceding context
            if 'context' in metadata:
                context_parts.append(f"Context: {metadata['context']}")
            
            # Add table type info
            if 'table_type' in metadata:
                context_parts.append(f"Table type: {', '.join(metadata['table_type'])}")
            
            # Add to text
            if context_parts:
                result['text'] = '\n'.join(context_parts) + '\n\n' + result['text']
        
        return results
    
    def delete_by_doc_id(self, doc_id: str) -> bool:
        """Delete all chunks for a document"""
        try:
            self.qdrant.delete(
                collection_name=settings.QDRANT_COLLECTION,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id)
                        )
                    ]
                )
            )
            logger.info(f"Deleted chunks for doc_id: {doc_id}")
            return True
        
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            info = self.qdrant.get_collection(settings.QDRANT_COLLECTION)
            return {
                "name": settings.QDRANT_COLLECTION,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Info retrieval failed: {e}")
            return {}
    
    def diagnose_storage(self, sample_size: int = 5):
        """
        Diagnostic tool to check what's actually stored.
        """
        logger.info("Running storage diagnostics...")
        
        try:
            # Get sample points
            scroll_result = self.qdrant.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                limit=sample_size,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            
            for i, point in enumerate(points):
                logger.info(f"\n--- Sample Chunk {i+1} ---")
                logger.info(f"ID: {point.id}")
                logger.info(f"Page: {point.payload.get('page_number')}")
                logger.info(f"Type: {point.payload.get('element_type')}")
                logger.info(f"Text length: {len(point.payload.get('text', ''))}")
                logger.info(f"Contains numbers: {any(c.isdigit() for c in point.payload.get('text', ''))}")
                logger.info(f"First 200 chars: {point.payload.get('text', '')[:200]}")
                
                if point.payload.get('element_type') == 'table':
                    logger.info(f"Table metadata: {point.payload.get('table_type')}")
                    logger.info(f"Keywords: {point.payload.get('searchable_keywords', '')[:100]}")
        
        except Exception as e:
            logger.error(f"Diagnostics failed: {e}")


# Singleton instance - MUST be at module level for import
embeddings_store = EnhancedEmbeddingsStore()


# Convenience functions
def upsert_chunks(chunks: List[Dict[str, Any]]) -> int:
    return embeddings_store.upsert_chunks(chunks)

def search(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    return embeddings_store.search_with_context(query, top_k)

# Export for clarity
__all__ = ['EnhancedEmbeddingsStore', 'embeddings_store', 'upsert_chunks', 'search']