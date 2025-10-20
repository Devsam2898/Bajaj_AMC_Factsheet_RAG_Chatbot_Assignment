"""
Simple Gradio Frontend for AMC Factsheet Chatbot
"""

import gradio as gr
import requests

# Your Modal API URL
API_URL = "https://devsam2898--amc-factsheet-rag-fastapi-app.modal.run/query"

# Available PDFs
FACTSHEETS = [
    "bajaj_finserv_factsheet.pdf",
]

def chat(message, history, selected_pdf):
    """Send message to API and get response"""
    try:
        response = requests.post(
            API_URL,
            json={"query": message},
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("answer", "No answer found")
        else:
            return f"Error: Could not get response"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Create interface
with gr.Blocks() as demo:
    gr.Markdown("# Bajaj AMC Factsheet Assistant")
    
    with gr.Row():
        # Dropdown for PDF selection
        pdf_dropdown = gr.Dropdown(
            choices=FACTSHEETS,
            value=FACTSHEETS[0],
            label="Select Factsheet"
        )
    
    # Chatbot
    chatbot = gr.ChatInterface(
        chat,
        additional_inputs=[pdf_dropdown],
        title="",
        description="Ask questions about the factsheet"
    )

# Run
if __name__ == "__main__":
    demo.launch()