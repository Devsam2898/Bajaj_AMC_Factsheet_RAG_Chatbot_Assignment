"""
Enhanced PDF parsing for AMC factsheets with intelligent table handling.
Extracts and enriches tables with searchable metadata.
"""

import fitz  # PyMuPDF
import camelot
import tabula
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None

from template.config import settings
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParsedElement:
    """Structured representation of parsed PDF elements"""
    text: str
    element_type: str  # 'text', 'table', 'header', 'chart_text'
    page_number: int
    bbox: tuple = None
    metadata: Dict[str, Any] = None


class EnhancedPDFParser:
    """
    Advanced PDF parser with table intelligence for AMC factsheets.
    """
    
    # Financial terms for metadata enrichment
    FINANCIAL_KEYWORDS = {
        'returns': ['return', 'returns', 'CAGR', 'performance', 'yield'],
        'allocation': ['allocation', 'holding', 'holdings', 'portfolio', '%', 'weight'],
        'ratios': ['ratio', 'expense', 'NAV', 'alpha', 'beta', 'sharpe'],
        'fund_info': ['fund manager', 'CEO', 'AUM', 'inception', 'benchmark'],
        'companies': ['HDFC', 'Reliance', 'TCS', 'Infosys', 'ICICI', 'bank', 'limited']
    }
    
    def __init__(self):
        self.ocr = None
        if settings.USE_OCR and PADDLEOCR_AVAILABLE:
            try:
                self.ocr = PaddleOCR(use_angle_cls=True, lang=settings.OCR_LANG, show_log=False)
                logger.info("✅ PaddleOCR initialized")
            except Exception as e:
                logger.warning(f"⚠️ OCR unavailable: {e}")
    
    def parse(self, pdf_path: str) -> List[ParsedElement]:
        """
        Main parsing with intelligent table handling.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"📄 Parsing: {pdf_path.name}")
        
        elements = []
        
        # Extract text with context awareness
        text_elements = self._extract_text_with_context(pdf_path)
        elements.extend(text_elements)
        
        # Extract tables with enrichment
        table_elements = self._extract_enriched_tables(pdf_path)
        elements.extend(table_elements)
        
        # OCR if available
        if self.ocr:
            ocr_elements = self._extract_ocr_text(pdf_path)
            elements.extend(ocr_elements)
        
        # Sort by page and position
        elements.sort(key=lambda x: (x.page_number, x.bbox[1] if x.bbox else 0))
        
        logger.info(f"✅ Extracted {len(elements)} elements ({sum(1 for e in elements if e.element_type == 'table')} tables)")
        return elements
    
    def _extract_text_with_context(self, pdf_path: Path) -> List[ParsedElement]:
        """Extract text with preceding headers as context"""
        elements = []
        doc = fitz.open(pdf_path)
        
        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("dict")["blocks"]
            last_header = ""
            
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    text = self._extract_block_text(block)
                    
                    if not text or len(text) < 10:
                        continue
                    
                    is_header = self._is_header(block)
                    font_size = self._get_font_size(block)
                    
                    if is_header:
                        last_header = text
                    
                    # Enrich metadata with context
                    metadata = {
                        'font_size': font_size,
                        'preceding_header': last_header,
                        'financial_categories': self._identify_financial_categories(text)
                    }
                    
                    elements.append(ParsedElement(
                        text=text,
                        element_type='header' if is_header else 'text',
                        page_number=page_num,
                        bbox=block.get("bbox"),
                        metadata=metadata
                    ))
        
        doc.close()
        logger.info(f"📝 Extracted {len(elements)} text blocks")
        return elements
    
    def _extract_enriched_tables(self, pdf_path: Path) -> List[ParsedElement]:
        """
        Extract tables with intelligent enrichment and context.
        """
        elements = []
        
        # Try Camelot first (best for bordered tables)
        camelot_tables = self._extract_tables_camelot(pdf_path)
        
        # Process each table with enrichment
        for table_elem in camelot_tables:
            enriched = self._enrich_table(table_elem)
            elements.append(enriched)
        
        # Fallback to Tabula if no tables found
        if not camelot_tables:
            tabula_tables = self._extract_tables_tabula(pdf_path)
            for table_elem in tabula_tables:
                enriched = self._enrich_table(table_elem)
                elements.append(enriched)
        
        return elements
    
    def _enrich_table(self, table_elem: ParsedElement) -> ParsedElement:
        """
        Add searchable metadata and context to tables.
        """
        text = table_elem.text
        
        # Parse table back to DataFrame for analysis
        try:
            # Extract table from markdown
            lines = [line.strip() for line in text.split('\n') if line.strip() and not line.startswith('|---')]
            
            if len(lines) < 2:
                return table_elem
            
            # Parse markdown table
            rows = []
            for line in lines:
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                rows.append(cells)
            
            if not rows:
                return table_elem
                
            df = pd.DataFrame(rows[1:], columns=rows[0])
            
            # Extract key information
            metadata = table_elem.metadata or {}
            metadata.update(self._analyze_table_content(df, text))
            
            # Create enriched text with context
            enriched_text = self._create_enriched_table_text(df, text, metadata)
            
            table_elem.text = enriched_text
            table_elem.metadata = metadata
            
        except Exception as e:
            logger.warning(f"⚠️ Table enrichment failed: {e}")
        
        return table_elem
    
    def _analyze_table_content(self, df: pd.DataFrame, original_text: str) -> Dict[str, Any]:
        """
        Analyze table to extract searchable metadata.
        """
        metadata = {
            'table_type': [],
            'contains_numbers': False,
            'contains_percentages': False,
            'fund_names': [],
            'company_names': [],
            'years': [],
            'key_metrics': []
        }
        
        full_text = original_text.lower()
        
        # Detect table type
        if any(term in full_text for term in ['return', 'performance', 'cagr']):
            metadata['table_type'].append('returns')
        if any(term in full_text for term in ['holding', 'allocation', 'portfolio']):
            metadata['table_type'].append('holdings')
        if any(term in full_text for term in ['expense', 'ratio', 'nav', 'aum']):
            metadata['table_type'].append('fund_metrics')
        if any(term in full_text for term in ['manager', 'team', 'ceo']):
            metadata['table_type'].append('personnel')
        
        # Extract numerical patterns
        if re.search(r'\d+\.?\d*%', original_text):
            metadata['contains_percentages'] = True
        if re.search(r'\d+\.?\d+', original_text):
            metadata['contains_numbers'] = True
        
        # Extract years
        years = re.findall(r'\b(20\d{2}|19\d{2})\b', original_text)
        metadata['years'] = list(set(years))
        
        # Extract fund names
        fund_patterns = [
            r'Bajaj\s+\w+(?:\s+\w+)*\s+Fund',
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Fund'
        ]
        for pattern in fund_patterns:
            funds = re.findall(pattern, original_text)
            metadata['fund_names'].extend(funds)
        
        # Extract company names (from holdings)
        for category, keywords in self.FINANCIAL_KEYWORDS.items():
            if category == 'companies':
                for keyword in keywords:
                    if keyword.lower() in full_text:
                        metadata['company_names'].append(keyword)
        
        # Extract key metrics (columns that might be important)
        for col in df.columns:
            col_lower = str(col).lower()
            if any(term in col_lower for term in ['%', 'return', 'nav', 'ratio', 'allocation']):
                metadata['key_metrics'].append(str(col))
        
        return metadata
    
    def _create_enriched_table_text(self, df: pd.DataFrame, original_text: str, metadata: Dict) -> str:
        """
        Create searchable text representation of table with context.
        """
        parts = []
        
        # Add table type context
        if metadata.get('table_type'):
            parts.append(f"TABLE TYPE: {', '.join(metadata['table_type'])}")
        
        # Add fund context
        if metadata.get('fund_names'):
            parts.append(f"FUNDS: {', '.join(set(metadata['fund_names']))}")
        
        # Add company context (for holdings)
        if metadata.get('company_names'):
            parts.append(f"COMPANIES: {', '.join(set(metadata['company_names'][:10]))}")
        
        # Add the actual table in markdown
        parts.append("\nTABLE DATA:")
        parts.append(original_text)
        
        # Add row-by-row text representation for better searchability
        parts.append("\nSEARCHABLE FORMAT:")
        for idx, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            if row_text.strip():
                parts.append(row_text)
        
        return "\n".join(parts)
    
    def _extract_tables_camelot(self, pdf_path: Path) -> List[ParsedElement]:
        """Extract tables using Camelot"""
        elements = []
        
        try:
            tables = camelot.read_pdf(
                str(pdf_path),
                pages='all',
                flavor='lattice',
                suppress_stdout=True
            )
            
            for table in tables:
                df = table.df
                
                # Skip empty or tiny tables
                if df.empty or df.shape[0] < 2 or df.shape[1] < 2:
                    continue
                
                # Clean DataFrame
                df = self._clean_dataframe(df)
                
                markdown_table = df.to_markdown(index=False)
                
                elements.append(ParsedElement(
                    text=markdown_table,
                    element_type='table',
                    page_number=table.page,
                    bbox=None,
                    metadata={
                        'accuracy': table.accuracy,
                        'shape': df.shape,
                        'source': 'camelot'
                    }
                ))
            
            logger.info(f"📊 Extracted {len(elements)} tables (Camelot)")
        
        except Exception as e:
            logger.warning(f"⚠️ Camelot failed: {e}")
        
        return elements
    
    def _extract_tables_tabula(self, pdf_path: Path) -> List[ParsedElement]:
        """Fallback table extraction using Tabula"""
        elements = []
        
        try:
            tables = tabula.read_pdf(
                str(pdf_path),
                pages='all',
                multiple_tables=True,
                silent=True
            )
            
            for idx, df in enumerate(tables):
                if df.empty or df.shape[0] < 2:
                    continue
                
                df = self._clean_dataframe(df)
                markdown_table = df.to_markdown(index=False)
                
                elements.append(ParsedElement(
                    text=markdown_table,
                    element_type='table',
                    page_number=idx + 1,
                    bbox=None,
                    metadata={
                        'shape': df.shape,
                        'source': 'tabula'
                    }
                ))
            
            logger.info(f"📊 Extracted {len(elements)} tables (Tabula)")
        
        except Exception as e:
            logger.warning(f"⚠️ Tabula failed: {e}")
        
        return elements
    
    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize DataFrame"""
        # Remove completely empty rows/columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Clean cell values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        
        return df
    
    def _extract_ocr_text(self, pdf_path: Path) -> List[ParsedElement]:
        """Extract text from images using OCR"""
        elements = []
        doc = fitz.open(pdf_path)
        
        for page_num, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")
            
            try:
                result = self.ocr.ocr(img_bytes, cls=True)
                
                if result and result[0]:
                    ocr_text = " ".join([line[1][0] for line in result[0]])
                    
                    if ocr_text.strip():
                        elements.append(ParsedElement(
                            text=ocr_text.strip(),
                            element_type='chart_text',
                            page_number=page_num,
                            bbox=None,
                            metadata={'source': 'ocr'}
                        ))
            
            except Exception as e:
                logger.warning(f"⚠️ OCR failed on page {page_num}: {e}")
        
        doc.close()
        return elements
    
    @staticmethod
    def _extract_block_text(block: dict) -> str:
        """Extract text from block"""
        text = ""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text += span.get("text", "") + " "
        return text.strip()
    
    @staticmethod
    def _is_header(block: dict) -> bool:
        """Detect headers by font size"""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("size", 0) > 14:
                    return True
        return False
    
    @staticmethod
    def _get_font_size(block: dict) -> float:
        """Get average font size"""
        sizes = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sizes.append(span.get("size", 12))
        return sum(sizes) / len(sizes) if sizes else 12.0
    
    def _identify_financial_categories(self, text: str) -> List[str]:
        """Identify which financial categories a text belongs to"""
        text_lower = text.lower()
        categories = []
        
        for category, keywords in self.FINANCIAL_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                categories.append(category)
        
        return categories


# Singleton instance - MUST be at module level for import
pdf_parser = EnhancedPDFParser()

# Export for clarity
__all__ = ['EnhancedPDFParser', 'ParsedElement', 'pdf_parser']


# # """
# # Local PDF parsing for AMC factsheets.
# # NO external APIs - fully secure for banking applications.
# # """

# # import fitz  # PyMuPDF
# # import camelot
# # import tabula
# # from pathlib import Path
# # from typing import List, Dict, Any
# # from dataclasses import dataclass

# # # Make PaddleOCR optional (not installed in Modal)
# # try:
# #     from paddleocr import PaddleOCR
# #     PADDLEOCR_AVAILABLE = True
# # except ImportError:
# #     PADDLEOCR_AVAILABLE = False
# #     PaddleOCR = None

# # from template.config import settings
# # import logging

# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)


# # @dataclass
# # class ParsedElement:
# #     """Structured representation of parsed PDF elements"""
# #     text: str
# #     element_type: str  # 'text', 'table', 'header', 'chart_text'
# #     page_number: int
# #     bbox: tuple = None  # (x0, y0, x1, y1) coordinates
# #     metadata: Dict[str, Any] = None


# # class LocalPDFParser:
# #     """
# #     Multi-strategy PDF parser optimized for AMC factsheets.
# #     Uses PyMuPDF for text, Camelot/Tabula for tables, PaddleOCR for images (optional).
# #     """
    
# #     def __init__(self):
# #         self.ocr = None
# #         if settings.USE_OCR and PADDLEOCR_AVAILABLE:
# #             try:
# #                 self.ocr = PaddleOCR(
# #                     use_angle_cls=True, 
# #                     lang=settings.OCR_LANG,
# #                     show_log=False
# #                 )
# #                 logger.info("✅ PaddleOCR initialized")
# #             except Exception as e:
# #                 logger.warning(f"⚠️ OCR unavailable: {e}")
# #         elif settings.USE_OCR and not PADDLEOCR_AVAILABLE:
# #             logger.info("ℹ️ PaddleOCR not installed, OCR features disabled")
    
# #     def parse(self, pdf_path: str) -> List[ParsedElement]:
# #         """
# #         Main parsing orchestrator.
# #         Returns structured elements preserving document layout.
# #         """
# #         pdf_path = Path(pdf_path)
# #         if not pdf_path.exists():
# #             raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
# #         logger.info(f"📄 Parsing: {pdf_path.name}")
        
# #         elements = []
        
# #         # Step 1: Extract text blocks with PyMuPDF
# #         text_elements = self._extract_text(pdf_path)
# #         elements.extend(text_elements)
        
# #         # Step 2: Extract tables with Camelot (lattice method)
# #         table_elements = self._extract_tables_camelot(pdf_path)
# #         elements.extend(table_elements)
        
# #         # Step 3: Fallback to Tabula for borderless tables
# #         if len(table_elements) == 0:
# #             table_elements = self._extract_tables_tabula(pdf_path)
# #             elements.extend(table_elements)
        
# #         # Step 4: OCR for chart labels and images (if enabled and available)
# #         if self.ocr:
# #             ocr_elements = self._extract_ocr_text(pdf_path)
# #             elements.extend(ocr_elements)
        
# #         # Sort by page and position
# #         elements.sort(key=lambda x: (x.page_number, x.bbox[1] if x.bbox else 0))
        
# #         logger.info(f"✅ Extracted {len(elements)} elements")
# #         return elements
    
# #     def _extract_text(self, pdf_path: Path) -> List[ParsedElement]:
# #         """Extract text blocks using PyMuPDF"""
# #         elements = []
# #         doc = fitz.open(pdf_path)
        
# #         for page_num, page in enumerate(doc, start=1):
# #             blocks = page.get_text("dict")["blocks"]
            
# #             for block in blocks:
# #                 if block.get("type") == 0:  # Text block
# #                     text = ""
# #                     for line in block.get("lines", []):
# #                         for span in line.get("spans", []):
# #                             text += span.get("text", "") + " "
                    
# #                     text = text.strip()
# #                     if not text or len(text) < 10:  # Skip tiny fragments
# #                         continue
                    
# #                     # Detect headers (larger font size)
# #                     is_header = self._is_header(block)
                    
# #                     elements.append(ParsedElement(
# #                         text=text,
# #                         element_type='header' if is_header else 'text',
# #                         page_number=page_num,
# #                         bbox=block.get("bbox"),
# #                         metadata={'font_size': self._get_font_size(block)}
# #                     ))
        
# #         doc.close()
# #         logger.info(f"📝 Extracted {len(elements)} text blocks")
# #         return elements
    
# #     def _extract_tables_camelot(self, pdf_path: Path) -> List[ParsedElement]:
# #         """Extract tables using Camelot (best for bordered tables)"""
# #         elements = []
        
# #         try:
# #             # Lattice mode for bordered tables (common in factsheets)
# #             tables = camelot.read_pdf(
# #                 str(pdf_path),
# #                 pages='all',
# #                 flavor='lattice',
# #                 suppress_stdout=True
# #             )
            
# #             for table in tables:
# #                 df = table.df
                
# #                 # Convert to markdown format
# #                 markdown_table = df.to_markdown(index=False)
                
# #                 elements.append(ParsedElement(
# #                     text=markdown_table,
# #                     element_type='table',
# #                     page_number=table.page,
# #                     bbox=None,
# #                     metadata={
# #                         'accuracy': table.accuracy,
# #                         'shape': df.shape
# #                     }
# #                 ))
            
# #             logger.info(f"📊 Extracted {len(elements)} tables (Camelot)")
        
# #         except Exception as e:
# #             logger.warning(f"⚠️ Camelot failed: {e}")
        
# #         return elements
    
# #     def _extract_tables_tabula(self, pdf_path: Path) -> List[ParsedElement]:
# #         """Fallback table extraction using Tabula (borderless tables)"""
# #         elements = []
        
# #         try:
# #             tables = tabula.read_pdf(
# #                 str(pdf_path),
# #                 pages='all',
# #                 multiple_tables=True,
# #                 silent=True
# #             )
            
# #             for idx, df in enumerate(tables):
# #                 if df.empty or df.shape[0] < 2:  # Skip tiny tables
# #                     continue
                
# #                 markdown_table = df.to_markdown(index=False)
                
# #                 elements.append(ParsedElement(
# #                     text=markdown_table,
# #                     element_type='table',
# #                     page_number=idx + 1,  # Approximate
# #                     bbox=None,
# #                     metadata={'shape': df.shape}
# #                 ))
            
# #             logger.info(f"📊 Extracted {len(elements)} tables (Tabula)")
        
# #         except Exception as e:
# #             logger.warning(f"⚠️ Tabula failed: {e}")
        
# #         return elements
    
# #     def _extract_ocr_text(self, pdf_path: Path) -> List[ParsedElement]:
# #         """Extract text from images/charts using OCR"""
# #         elements = []
# #         doc = fitz.open(pdf_path)
        
# #         for page_num, page in enumerate(doc, start=1):
# #             # Get page as image
# #             pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution
# #             img_bytes = pix.tobytes("png")
            
# #             # Run OCR
# #             try:
# #                 result = self.ocr.ocr(img_bytes, cls=True)
                
# #                 if result and result[0]:
# #                     ocr_text = " ".join([line[1][0] for line in result[0]])
                    
# #                     if ocr_text.strip():
# #                         elements.append(ParsedElement(
# #                             text=ocr_text.strip(),
# #                             element_type='chart_text',
# #                             page_number=page_num,
# #                             bbox=None,
# #                             metadata={'source': 'ocr'}
# #                         ))
            
# #             except Exception as e:
# #                 logger.warning(f"⚠️ OCR failed on page {page_num}: {e}")
        
# #         doc.close()
# #         logger.info(f"🔍 Extracted {len(elements)} OCR elements")
# #         return elements
    
# #     @staticmethod
# #     def _is_header(block: dict) -> bool:
# #         """Detect if text block is a header based on font size"""
# #         for line in block.get("lines", []):
# #             for span in line.get("spans", []):
# #                 font_size = span.get("size", 0)
# #                 if font_size > 14:  # Headers typically > 14pt
# #                     return True
# #         return False
    
# #     @staticmethod
# #     def _get_font_size(block: dict) -> float:
# #         """Get average font size from block"""
# #         sizes = []
# #         for line in block.get("lines", []):
# #             for span in line.get("spans", []):
# #                 sizes.append(span.get("size", 12))
# #         return sum(sizes) / len(sizes) if sizes else 12.0


# # # Singleton instance
# # pdf_parser = LocalPDFParser()