"""
LlamaParse integration for accurate AMC factsheet parsing.
Replaces local PDF parsing with cloud-based intelligence.
"""

from llama_parse import LlamaParse
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import re

# Fix for nested event loops
import nest_asyncio
nest_asyncio.apply()

logger = logging.getLogger(__name__)


@dataclass
class ParsedElement:
    """Structured representation of parsed PDF elements"""
    text: str
    element_type: str  # 'text', 'table', 'header'
    page_number: int
    bbox: tuple = None
    metadata: Dict[str, Any] = None


class LlamaParseHandler:
    """
    High-accuracy PDF parsing using LlamaParse.
    Optimized for financial documents with complex tables.
    """
    
    # Financial terms for metadata enrichment
    FINANCIAL_KEYWORDS = {
        'returns': ['return', 'returns', 'CAGR', 'performance', 'yield'],
        'allocation': ['allocation', 'holding', 'holdings', 'portfolio', '%', 'weight'],
        'ratios': ['ratio', 'expense', 'NAV', 'alpha', 'beta', 'sharpe'],
        'fund_info': ['fund manager', 'CEO', 'AUM', 'inception', 'benchmark'],
        'companies': ['HDFC', 'Reliance', 'TCS', 'Infosys', 'ICICI', 'bank', 'limited']
    }
    
    def __init__(self, api_key: str):
        """
        Initialize LlamaParse with optimized settings for factsheets.
        
        Args:
            api_key: LlamaParse API key
        """
        self.parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",  # Get markdown with table structure preserved
            verbose=True,
            language="en",
            num_workers=4,  # Parallel processing
            # Optimize for tables
            parsing_instruction="""
            This is an AMC (Asset Management Company) mutual fund factsheet containing:
            - Financial performance tables with returns, NAV, and ratios
            - Portfolio holdings with company names and allocation percentages
            - Fund manager information 
            - Expense ratios and other key metrics
            
            Please:
            1. Preserve all table structures exactly
            2. Keep numerical values precise (percentages, decimals, dates)
            3. Maintain relationships between headers and data
            4. Extract all company names and fund names accurately
            5. Preserve column headers and row labels
            """,
        )
        logger.info("LlamaParse initialized")
    
    def parse(self, pdf_path: str) -> List[ParsedElement]:
        """
        Parse PDF using LlamaParse and structure the results.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of ParsedElement objects
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Parsing with LlamaParse: {pdf_path.name}")
        
        # Parse PDF
        documents = self.parser.load_data(str(pdf_path))
        
        logger.info(f"LlamaParse extracted {len(documents)} documents")
        
        # Convert to structured elements
        elements = self._structure_documents(documents)
        
        logger.info(f"Structured into {len(elements)} elements")
        return elements
    
    def _structure_documents(self, documents) -> List[ParsedElement]:
        """
        Convert LlamaParse documents into structured elements.
        """
        elements = []
        
        for doc in documents:
            # Get page number from metadata
            page_num = doc.metadata.get('page_number', 1)
            
            # Split markdown into sections
            sections = self._split_markdown(doc.text)
            
            for section in sections:
                # Detect element type
                element_type = self._detect_element_type(section['text'])
                
                # Extract metadata
                metadata = self._extract_metadata(section['text'], element_type)
                
                elements.append(ParsedElement(
                    text=section['text'],
                    element_type=element_type,
                    page_number=page_num,
                    bbox=None,
                    metadata=metadata
                ))
        
        return elements
    
    def _split_markdown(self, markdown_text: str) -> List[Dict[str, str]]:
        """
        Split markdown into logical sections (headers, tables, text).
        """
        sections = []
        current_section = []
        in_table = False
        
        lines = markdown_text.split('\n')
        
        for line in lines:
            # Detect table boundaries
            if '|' in line and not in_table:
                # Start of table
                if current_section:
                    sections.append({'text': '\n'.join(current_section).strip()})
                    current_section = []
                in_table = True
                current_section.append(line)
            
            elif in_table:
                if '|' in line or line.startswith('|---') or not line.strip():
                    current_section.append(line)
                else:
                    # End of table
                    sections.append({'text': '\n'.join(current_section).strip()})
                    current_section = [line]
                    in_table = False
            
            # Detect headers (lines starting with #)
            elif line.startswith('#'):
                if current_section:
                    sections.append({'text': '\n'.join(current_section).strip()})
                sections.append({'text': line.strip()})
                current_section = []
            
            else:
                current_section.append(line)
        
        # Add remaining content
        if current_section:
            sections.append({'text': '\n'.join(current_section).strip()})
        
        return [s for s in sections if s['text']]
    
    def _detect_element_type(self, text: str) -> str:
        """
        Detect if text is a table, header, or regular text.
        """
        # Check for table
        if '|' in text and text.count('|') > 2:
            return 'table'
        
        # Check for header
        if text.startswith('#') or (len(text) < 100 and text.isupper()):
            return 'header'
        
        return 'text'
    
    def _extract_metadata(self, text: str, element_type: str) -> Dict[str, Any]:
        """
        Extract rich metadata from text for better retrieval.
        """
        metadata = {
            'char_count': len(text),
            'financial_categories': self._identify_financial_categories(text)
        }
        
        # Enhanced metadata for tables
        if element_type == 'table':
            metadata.update(self._analyze_table_metadata(text))
        
        return metadata
    
    def _analyze_table_metadata(self, table_text: str) -> Dict[str, Any]:
        """
        Extract metadata from table content.
        """
        metadata = {
            'table_type': [],
            'contains_numbers': bool(re.search(r'\d+\.?\d*', table_text)),
            'contains_percentages': '%' in table_text,
            'fund_names': [],
            'company_names': [],
            'years': [],
            'key_metrics': []
        }
        
        text_lower = table_text.lower()
        
        # Detect table type
        if any(term in text_lower for term in ['return', 'performance', 'cagr', 'yield']):
            metadata['table_type'].append('returns')
        if any(term in text_lower for term in ['holding', 'allocation', 'portfolio']):
            metadata['table_type'].append('holdings')
        if any(term in text_lower for term in ['expense', 'ratio', 'nav', 'aum']):
            metadata['table_type'].append('fund_metrics')
        if any(term in text_lower for term in ['manager', 'team', 'ceo', 'cio']):
            metadata['table_type'].append('personnel')
        
        # Extract fund names
        fund_patterns = [
            r'Bajaj\s+\w+(?:\s+\w+)*\s+Fund',
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Fund'
        ]
        for pattern in fund_patterns:
            funds = re.findall(pattern, table_text)
            metadata['fund_names'].extend(funds)
        
        # Extract company names
        company_patterns = [
            r'[A-Z][A-Za-z]+\s+(?:Bank|Ltd|Limited|Industries|Corp|Company)',
            r'(?:HDFC|Reliance|TCS|Infosys|ICICI|Axis|Kotak)'
        ]
        for pattern in company_patterns:
            companies = re.findall(pattern, table_text)
            metadata['company_names'].extend(companies)
        
        # Extract years
        years = re.findall(r'\b(20\d{2}|19\d{2})\b', table_text)
        metadata['years'] = list(set(years))
        
        # Extract key metrics from table
        metric_terms = ['NAV', 'return', 'expense ratio', 'allocation', '%', 'AUM', 'alpha', 'beta']
        for term in metric_terms:
            if term.lower() in text_lower:
                metadata['key_metrics'].append(term)
        
        return metadata
    
    def _identify_financial_categories(self, text: str) -> List[str]:
        """
        Identify which financial categories a text belongs to.
        """
        text_lower = text.lower()
        categories = []
        
        for category, keywords in self.FINANCIAL_KEYWORDS.items():
            if any(keyword.lower() in text_lower for keyword in keywords):
                categories.append(category)
        
        return categories


# Singleton-style factory function
def create_llamaparse_parser(api_key: str) -> LlamaParseHandler:
    """
    Create LlamaParse parser instance.
    
    Args:
        api_key: LlamaParse API key
        
    Returns:
        LlamaParseHandler instance
    """
    return LlamaParseHandler(api_key=api_key)