import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT

def _create_pdf_from_text(text: str, filename: str, title: str = "") -> str:
    """
    Helper function to generate a clean professional PDF from formatted text.
    Uses ReportLab Platypus for flowable layout.
    """
    # Ensure the directory exists (e.g., if writing to data/ or assets/)
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Initialize document template
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    # Setup styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    normal_style = styles["Normal"]
    normal_style.fontSize = 11
    normal_style.leading = 14
    
    title_style = styles["Heading1"]
    title_style.alignment = TA_LEFT
    title_style.spaceAfter = 20
    
    Story = []
    
    # Add an optional title if needed (often the text itself includes a title but just in case)
    if title:
        Story.append(Paragraph(title, title_style))
    
    # Process text into paragraphs
    # Handle basic markdown/formatting artifacts if any (like <s>, [INST], **, etc.)
    # For a robust PDF generator, we split by newlines and handle them
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            # Add a visual spacer for empty lines
            Story.append(Spacer(1, 10))
            continue
            
        # Clean up some common markdown markers from LLM output for the PDF
        line = line.replace('**', '')
        line = line.replace('<s>', '')
        line = line.replace('[INST]', '')
        line = line.replace('[/INST]', '')
        line = line.replace('</s>', '')
        
        # Replace literal newline characters with br tags if any remain
        p_text = line.replace('\n', '<br />')
        
        # Create a paragraph and add it
        p = Paragraph(p_text, normal_style)
        Story.append(p)
        Story.append(Spacer(1, 4)) # Small space after each paragraph
        
    # Build the PDF
    doc.build(Story)
    
    return os.path.abspath(filename)


def create_resume_pdf(text: str, filename: str) -> str:
    """
    Takes generated resume text and creates a professional PDF.
    
    Args:
        text (str): The formatted resume text (usually from LLM)
        filename (str): The local path where the PDF should be saved
        
    Returns:
        str: Absolute path to the saved PDF
    """
    # Simply delegate to the helper
    return _create_pdf_from_text(text, filename)


def create_cover_letter_pdf(text: str, filename: str) -> str:
    """
    Takes generated cover letter text and creates a professional PDF.
    
    Args:
        text (str): The formatted cover letter text
        filename (str): The local path where the PDF should be saved
        
    Returns:
        str: Absolute path to the saved PDF
    """
    # Simply delegate to the helper
    return _create_pdf_from_text(text, filename)


def create_portfolio_pdf(text: str, filename: str) -> str:
    """
    Takes generated portfolio text and creates a professional PDF.
    
    Args:
        text (str): The formatted portfolio text
        filename (str): The local path where the PDF should be saved
        
    Returns:
        str: Absolute path to the saved PDF
    """
    # Simply delegate to the helper
    return _create_pdf_from_text(text, filename)
