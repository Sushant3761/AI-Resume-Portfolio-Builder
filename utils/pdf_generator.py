import re
from fpdf import FPDF

class ATSResumePDF(FPDF):
    def footer(self):
        # Quiet footer to preserve ATS compliance and page layout
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Generated via AI Platform | Page {self.page_no()}", align="C")

def compile_pdf(markdown_text: str) -> bytes:
    """
    Parses Markdown and compiles it into an ATS-friendly, highly structured PDF document.
    """
    pdf = ATSResumePDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # 0.75-inch margins (19mm)
    pdf.set_margins(left=19, top=19, right=19)
    
    pdf.set_font("Helvetica", size=10)
    
    lines = markdown_text.split("\n")
    
    for line in lines:
        line = line.strip()
        
        # Heading 1 (e.g. Applicant Name)
        if line.startswith("# "):
            val = line[2:].strip()
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(15, 23, 42) # Deep slate
            pdf.cell(0, 10, val, ln=True, align="C")
            pdf.ln(1)
            
        # Heading 2 (e.g. Section Headers)
        elif line.startswith("## "):
            val = line[3:].strip()
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 11.5)
            pdf.set_text_color(79, 70, 229) # Indigo accent
            pdf.cell(0, 8, val, ln=True)
            # Add a subtle horizontal line under header for visual hierarchy
            pdf.set_draw_color(226, 232, 240) # light gray
            pdf.set_line_width(0.5)
            pdf.line(19, pdf.get_y(), 191, pdf.get_y())
            pdf.ln(1.5)
            
        # Heading 3
        elif line.startswith("### "):
            val = line[4:].strip()
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(0, 6, val, ln=True)
            
        # Bullet points
        elif line.startswith("* ") or line.startswith("- "):
            val = line[2:].strip()
            pdf.set_font("Helvetica", size=9.5)
            pdf.set_text_color(51, 65, 85) # body dark gray
            
            # Simple list indent: print bullet, then text block
            pdf.set_x(24) # indent bullet
            pdf.write(4.5, chr(149) + "  ") # bullet symbol (dot)
            
            # Strip markdown bold indicator for inline rendering
            clean_val = re.sub(r'\*\*(.*?)\*\*', r'\1', val)
            pdf.write(4.5, clean_val)
            pdf.ln(5)
            
        # Empty space
        elif not line:
            pdf.ln(1.5)
            
        # Normal body paragraph
        else:
            pdf.set_font("Helvetica", size=9.5)
            pdf.set_text_color(51, 65, 85)
            pdf.set_x(19) # normal margin
            # Cleanly handle markdown bold within text (e.g. **Title**)
            clean_line = re.sub(r'\*\*(.*?)\*\*', r'\1', line) # strip raw asterisks for clean rendering
            pdf.multi_cell(0, 4.5, clean_line)
            pdf.ln(0.5)
            
    # Output PDF as byte string
    return bytes(pdf.output())
