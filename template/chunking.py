"""
Enhanced chunking strategy for AMC factsheets.
Preserves table integrity with rich metadata for better retrieval.
"""

from typing import List, Dict, Any
from template.pdf_parser import ParsedElement
from template.config import settings
import re


class EnhancedChunker:
    """
    Context-aware chunking with table intelligence.
    """
    
    def __init__(
        self, 
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        # Increase chunk size for tables
        self.table_chunk_size = self.chunk_size * 3  # Tables need more space
    
    def chunk_elements(
        self, 
        elements: List[ParsedElement],
        doc_id: str
    ) -> List[Dict[str, Any]]:
        """
        Convert parsed elements into optimally-sized chunks with rich metadata.
        """
        chunks = []
        context_window = []  # Track recent elements for context
        
        for i, element in enumerate(elements):
            # Update context window (last 2 elements)
            context_window.append(element)
            if len(context_window) > 2:
                context_window.pop(0)
            
            # Get preceding context
            preceding_context = self._build_context(context_window[:-1])
            
            # Tables get special treatment
            if element.element_type == 'table':
                chunks.extend(
                    self._chunk_table(element, doc_id, preceding_context)
                )
            
            # Headers are kept intact with context
            elif element.element_type == 'header':
                chunks.append(self._create_chunk(
                    text=element.text,
                    element_type=element.element_type,
                    page_number=element.page_number,
                    doc_id=doc_id,
                    metadata=element.metadata,
                    preceding_context=preceding_context
                ))
            
            # Text blocks are split with overlap
            else:
                text_chunks = self._split_text_with_overlap(element.text)
                
                for chunk_text in text_chunks:
                    chunks.append(self._create_chunk(
                        text=chunk_text,
                        element_type=element.element_type,
                        page_number=element.page_number,
                        doc_id=doc_id,
                        metadata=element.metadata,
                        preceding_context=preceding_context
                    ))
        
        return chunks
    
    def _chunk_table(
        self,
        table_elem: ParsedElement,
        doc_id: str,
        preceding_context: str
    ) -> List[Dict[str, Any]]:
        """
        Smart table chunking: keep intact if small, split intelligently if large.
        """
        chunks = []
        table_text = table_elem.text
        
        # Small tables: keep intact
        if len(table_text) <= self.table_chunk_size:
            chunks.append(self._create_chunk(
                text=table_text,
                element_type='table',
                page_number=table_elem.page_number,
                doc_id=doc_id,
                metadata=table_elem.metadata,
                preceding_context=preceding_context
            ))
        
        # Large tables: split by sections but keep metadata
        else:
            sections = self._split_table_intelligently(table_text)
            
            for idx, section in enumerate(sections):
                # Enrich metadata for each section
                section_metadata = table_elem.metadata.copy() if table_elem.metadata else {}
                section_metadata['table_section'] = f"{idx+1}/{len(sections)}"
                
                chunks.append(self._create_chunk(
                    text=section,
                    element_type='table',
                    page_number=table_elem.page_number,
                    doc_id=doc_id,
                    metadata=section_metadata,
                    preceding_context=preceding_context
                ))
        
        return chunks
    
    def _split_table_intelligently(self, table_text: str) -> List[str]:
        """
        Split large tables while preserving structure.
        """
        sections = []
        lines = table_text.split('\n')
        
        # Find the header and metadata sections
        header_end = 0
        for i, line in enumerate(lines):
            if line.startswith('TABLE DATA:'):
                header_end = i
                break
        
        # Keep header with each section
        header_lines = lines[:header_end+1]
        data_lines = lines[header_end+1:]
        
        # Split data into sections
        current_section = header_lines.copy()
        current_size = sum(len(line) for line in current_section)
        
        for line in data_lines:
            line_size = len(line)
            
            if current_size + line_size > self.table_chunk_size and len(current_section) > len(header_lines):
                sections.append('\n'.join(current_section))
                current_section = header_lines.copy()
                current_size = sum(len(line) for line in current_section)
            
            current_section.append(line)
            current_size += line_size
        
        # Add remaining section
        if len(current_section) > len(header_lines):
            sections.append('\n'.join(current_section))
        
        return sections if sections else [table_text]
    
    def _split_text_with_overlap(self, text: str) -> List[str]:
        """
        Split text into chunks with sliding window overlap.
        """
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Keep last N sentences for overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk, 
                    self.chunk_overlap
                )
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in overlap_sentences)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    @staticmethod
    def _build_context(elements: List[ParsedElement]) -> str:
        """Build context string from preceding elements"""
        if not elements:
            return ""
        
        context_parts = []
        for elem in elements:
            if elem.element_type == 'header':
                context_parts.append(f"SECTION: {elem.text}")
            elif elem.metadata and 'preceding_header' in elem.metadata:
                context_parts.append(f"CONTEXT: {elem.metadata['preceding_header']}")
        
        return " | ".join(context_parts) if context_parts else ""
    
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def _get_overlap_sentences(sentences: List[str], overlap_size: int) -> List[str]:
        """Get last N characters worth of sentences for overlap"""
        overlap = []
        char_count = 0
        
        for sentence in reversed(sentences):
            if char_count + len(sentence) > overlap_size:
                break
            overlap.insert(0, sentence)
            char_count += len(sentence)
        
        return overlap
    
    @staticmethod
    def _create_chunk(
        text: str,
        element_type: str,
        page_number: int,
        doc_id: str,
        metadata: Dict = None,
        preceding_context: str = ""
    ) -> Dict[str, Any]:
        """Create chunk with enriched metadata for better retrieval"""
        # Merge metadata
        chunk_metadata = {
            "doc_id": doc_id,
            "page_number": page_number,
            "element_type": element_type,
            "char_count": len(text),
            **(metadata or {})
        }
        
        # Add context to metadata for filtering
        if preceding_context:
            chunk_metadata['context'] = preceding_context
        
        # Add searchable keywords for tables
        if element_type == 'table':
            chunk_metadata['searchable_keywords'] = _extract_table_keywords(text)
        
        return {
            "text": text,
            "meta": chunk_metadata
        }


def _extract_table_keywords(table_text: str) -> str:
    """Extract searchable keywords from table for better retrieval"""
    keywords = []
    
    # Extract fund names
    fund_matches = re.findall(r'Bajaj\s+\w+(?:\s+\w+)*\s+Fund', table_text)
    keywords.extend(fund_matches)
    
    # Extract company names
    company_matches = re.findall(r'[A-Z][A-Za-z]+\s+(?:Bank|Ltd|Limited|Industries|Corp)', table_text)
    keywords.extend(company_matches)
    
    # Extract percentages
    percentage_matches = re.findall(r'\d+\.?\d*%', table_text)
    keywords.extend(percentage_matches)
    
    # Extract metric terms
    metric_terms = ['return', 'nav', 'expense ratio', 'allocation', 'holding', 'aum', 'alpha', 'beta']
    for term in metric_terms:
        if term.lower() in table_text.lower():
            keywords.append(term)
    
    return ' | '.join(set(keywords))


# Singleton instance - MUST be at module level for import
chunker = EnhancedChunker()

# # """
# # Intelligent chunking strategy for AMC factsheets.
# # Preserves table integrity while respecting token limits.
# # """

# # from typing import List, Dict, Any
# # from template.pdf_parser import ParsedElement
# # from template.config import settings
# # import re


# # class SmartChunker:
# #     """
# #     Context-aware chunking that:
# #     - Keeps tables intact
# #     - Adds overlap for context continuity
# #     - Preserves semantic boundaries (headers, paragraphs)
# #     """
    
# #     def __init__(
# #         self, 
# #         chunk_size: int = None,
# #         chunk_overlap: int = None
# #     ):
# #         self.chunk_size = chunk_size or settings.CHUNK_SIZE
# #         self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    
# #     def chunk_elements(
# #         self, 
# #         elements: List[ParsedElement],
# #         doc_id: str
# #     ) -> List[Dict[str, Any]]:
# #         """
# #         Convert parsed elements into optimally-sized chunks.
# #         """
# #         chunks = []
        
# #         for element in elements:
# #             # Tables and headers are kept intact (no splitting)
# #             if element.element_type in ['table', 'header']:
# #                 chunks.append(self._create_chunk(
# #                     text=element.text,
# #                     element_type=element.element_type,
# #                     page_number=element.page_number,
# #                     doc_id=doc_id,
# #                     metadata=element.metadata
# #                 ))
            
# #             # Text blocks are split with overlap
# #             else:
# #                 text_chunks = self._split_text_with_overlap(element.text)
                
# #                 for chunk_text in text_chunks:
# #                     chunks.append(self._create_chunk(
# #                         text=chunk_text,
# #                         element_type=element.element_type,
# #                         page_number=element.page_number,
# #                         doc_id=doc_id,
# #                         metadata=element.metadata
# #                     ))
        
# #         return chunks
    
# #     def _split_text_with_overlap(self, text: str) -> List[str]:
# #         """
# #         Split text into chunks with sliding window overlap.
# #         """
# #         # Split by sentences for clean boundaries
# #         sentences = self._split_sentences(text)
        
# #         chunks = []
# #         current_chunk = []
# #         current_length = 0
        
# #         for sentence in sentences:
# #             sentence_length = len(sentence)
            
# #             # If adding this sentence exceeds chunk_size, save current chunk
# #             if current_length + sentence_length > self.chunk_size and current_chunk:
# #                 chunks.append(" ".join(current_chunk))
                
# #                 # Keep last N sentences for overlap
# #                 overlap_sentences = self._get_overlap_sentences(
# #                     current_chunk, 
# #                     self.chunk_overlap
# #                 )
# #                 current_chunk = overlap_sentences
# #                 current_length = sum(len(s) for s in overlap_sentences)
            
# #             current_chunk.append(sentence)
# #             current_length += sentence_length
        
# #         # Add remaining chunk
# #         if current_chunk:
# #             chunks.append(" ".join(current_chunk))
        
# #         return chunks
    
# #     @staticmethod
# #     def _split_sentences(text: str) -> List[str]:
# #         """Split text into sentences"""
# #         # Simple sentence splitter (can be enhanced with NLTK if needed)
# #         sentences = re.split(r'(?<=[.!?])\s+', text)
# #         return [s.strip() for s in sentences if s.strip()]
    
# #     @staticmethod
# #     def _get_overlap_sentences(sentences: List[str], overlap_size: int) -> List[str]:
# #         """Get last N characters worth of sentences for overlap"""
# #         overlap = []
# #         char_count = 0
        
# #         for sentence in reversed(sentences):
# #             if char_count + len(sentence) > overlap_size:
# #                 break
# #             overlap.insert(0, sentence)
# #             char_count += len(sentence)
        
# #         return overlap
    
# #     @staticmethod
# #     def _create_chunk(
# #         text: str,
# #         element_type: str,
# #         page_number: int,
# #         doc_id: str,
# #         metadata: Dict = None
# #     ) -> Dict[str, Any]:
# #         """Create standardized chunk dictionary"""
# #         return {
# #             "text": text,
# #             "meta": {
# #                 "doc_id": doc_id,
# #                 "page_number": page_number,
# #                 "element_type": element_type,
# #                 "char_count": len(text),
# #                 **(metadata or {})
# #             }
# #         }


# # # Singleton instance
# # chunker = SmartChunker()