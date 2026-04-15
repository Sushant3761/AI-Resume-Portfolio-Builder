import os
import re
import markdown
from xhtml2pdf import pisa
import logging

def _clean_filename(filename: str) -> str:
    """Removes special characters from filenames to prevent OS path issues."""
    filename = os.path.basename(filename)
    # allow words, hyphens, underscores, dots, and spaces
    clean = re.sub(r'[^\w\-_\. ]', '_', filename)
    return clean

def _create_pdf_from_markdown(markdown_text: str, filename: str) -> str:
    """
    Converts markdown text directly to HTML, then renders it to PDF using xhtml2pdf.
    Applies simplistic styling to guarantee layout safety.
    """
    # 1. Clean filename
    safe_filename = _clean_filename(filename)
    safe_path = os.path.abspath(safe_filename)
    os.makedirs(os.path.dirname(safe_path) if os.path.dirname(safe_path) else ".", exist_ok=True)
    
    # 2. Convert markdown to HTML HTML
    # We use extension 'tables' just in case, 'sane_lists' for strictly formatted lists
    html_content = markdown.markdown(markdown_text, extensions=['tables', 'sane_lists'])
    
    # 3. Wrap HTML in a simple template with baseline CSS
    # xhtml2pdf has limited CSS support, so we keep it very basic: standard font types, no flexbox
    full_html = f"""
    <html>
    <head>
        <style>
            @page {{
                size: letter portrait;
                margin: 2cm;
            }}
            body {{
                font-family: Helvetica, Arial, sans-serif;
                font-size: 11pt;
                line-height: 1.4;
                color: #333333;
            }}
            h1, h2, h3 {{
                color: #000000;
                margin-top: 15px;
                margin-bottom: 5px;
            }}
            h1 {{ font-size: 18pt; border-bottom: 1px solid #000; padding-bottom: 3px; }}
            h2 {{ font-size: 14pt; }}
            h3 {{ font-size: 12pt; }}
            ul {{
                margin-top: 5px;
                margin-bottom: 5px;
            }}
            li {{
                margin-bottom: 3px;
            }}
            p {{
                margin-top: 5px;
                margin-bottom: 5px;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # 4. Write PDF
    try:
        with open(safe_path, "w+b") as result_file:
            # convert HTML to PDF
            pisa_status = pisa.CreatePDF(
                full_html,
                dest=result_file
            )
            
            if pisa_status.err:
                logging.error("xhtml2pdf encountered an error generating the PDF")
                
    except Exception as e:
        logging.error("Failed to generate PDF: %s", str(e))
        
    return safe_path

def create_resume_pdf(text: str, filename: str) -> str:
    return _create_pdf_from_markdown(text, filename)

def create_cover_letter_pdf(text: str, filename: str) -> str:
    return _create_pdf_from_markdown(text, filename)

def create_portfolio_pdf(text: str, filename: str) -> str:
    return _create_pdf_from_markdown(text, filename)
