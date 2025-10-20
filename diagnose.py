"""
Re-ingest PDF to replace old chunks with enhanced ones.
Run this after deploying the enhanced parser/chunker.
"""

import requests
import sys

# Your Modal endpoint
MODAL_ENDPOINT = "https://devsam2898--amc-factsheet-rag-fastapi-app.modal.run"  # Replace with your actual endpoint

def delete_existing_data(doc_id: str = "bajaj_finserv_factsheet"):
    """Delete old chunks before re-ingesting"""
    print(f"Deleting existing data for doc_id: {doc_id}...")
    
    response = requests.delete(
        f"{MODAL_ENDPOINT}/delete/{doc_id}"
    )
    
    if response.status_code == 200:
        print(f"Deleted successfully")
        return True
    else:
        print(f"Delete failed: {response.text}")
        return False


def reingest_pdf(pdf_path: str):
    """Re-ingest PDF with enhanced parsing"""
    print(f"Re-ingesting PDF: {pdf_path}")
    
    with open(pdf_path, 'rb') as f:
        files = {'file': f}
        
        response = requests.post(
            f"{MODAL_ENDPOINT}/ingest",
            files=files
        )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Ingestion successful!")
        print(f"   - Document ID: {result.get('doc_id')}")
        print(f"   - Chunks created: {result.get('num_chunks')}")
        print(f"   - Processing time: {result.get('processing_time_ms')}ms")
        return result
    else:
        print(f"Ingestion failed: {response.text}")
        return None


def test_query(question: str):
    """Test a query after re-ingestion"""
    print(f"\nTesting query: {question}")
    
    response = requests.post(
        f"{MODAL_ENDPOINT}/query",
        json={"query": question}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Query successful!")
        print(f"\nAnswer: {result.get('answer')}")
        print(f"\nSources used: {result.get('num_sources')}")
        
        # Check citation quality
        citations = result.get('citations', [])
        for i, citation in enumerate(citations[:3], 1):
            print(f"\n--- Citation {i} ---")
            print(f"Page: {citation.get('page_number')}")
            print(f"Type: {citation.get('element_type')}")
            print(f"Char count: {citation.get('char_count')}")
            
            # This should now be much larger!
            if citation.get('char_count', 0) > 100:
                print(f"Good chunk size!")
            else:
                print(f" Still too small - reingest may not have worked")
        
        return result
    else:
        print(f"Query failed: {response.text}")
        return None


def run_full_reingest(pdf_path: str):
    """Complete re-ingestion pipeline"""
    print("STARTING RE-INGESTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Delete old data
    delete_existing_data()
    
    # Step 2: Re-ingest with enhanced parser
    result = reingest_pdf(pdf_path)
    
    if not result:
        print("Re-ingestion failed. Aborting.")
        return
    
    # Step 3: Test queries
    print("\n" + "=" * 60)
    print("TESTING QUERIES")
    print("=" * 60)
    
    test_questions = [
        "Give me the fund returns % of last 15 days for Bajaj Finserv Liquid Fund",
        "What is % of NAV for HDFC Bank in Bajaj Large Cap Fund?",
        "Who is the CEO of Bajaj AMC?",
        "What is Jensen’s Alpha ?"
    ]
    
    for question in test_questions:
        test_query(question)
        print("\n" + "-" * 60)
    
    print("\nRE-INGESTION PIPELINE COMPLETE!")


if __name__ == "__main__":
    from pathlib import Path
    
    # Update these values
    MODAL_ENDPOINT = "https://devsam2898--amc-factsheet-rag-fastapi-app.modal.run"  # YOUR ACTUAL ENDPOINT
    PDF_PATH = r"D:\Bajaj\data\amc_factsheets\bajaj_finserv_factsheet.pdf"  # Use raw string or Path
    
    if len(sys.argv) > 1:
        PDF_PATH = sys.argv[1]
    
    # Validate PDF path
    pdf_file = Path(PDF_PATH)
    if not pdf_file.exists():
        print(f"Error: PDF not found at {PDF_PATH}")
        print(f"   Tried: {pdf_file.absolute()}")
        sys.exit(1)
    
    if "your-app" in MODAL_ENDPOINT:
        print("Error: Update MODAL_ENDPOINT with your actual Modal URL")
        print("Example: https://dev--bajaj-rag-chatbot-fastapi-app.modal.run")
        sys.exit(1)
    
    run_full_reingest(str(pdf_file))