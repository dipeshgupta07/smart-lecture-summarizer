def generate_pdf(text, filename="Document_Summary.pdf"):
    """
    Generate a PDF file from text using FPDF, handling encoding issues 
    and returning the content as bytes using io.BytesIO (memory-based).
    """
    from fpdf import FPDF
    import io
    
    pdf = FPDF()
    pdf.add_page()
    
    # Use a standard font like Arial
    pdf.set_font("Arial", size=10)

    # Split text into lines
    lines = text.split("\n")
    
    for line in lines:
        if line.strip():
            # Remove any characters that can't be encoded in latin-1
            clean_line = line.encode('latin-1', 'replace').decode('latin-1')
            
            # multi_cell automatically wraps text.
            pdf.multi_cell(0, 6, clean_line)
        else:
            # Add spacing for empty lines
            pdf.ln(3)
    
    # Create an in-memory bytes buffer
    pdf_buffer = io.BytesIO()
    
    # Get PDF content as bytes and write to buffer
    pdf_output = pdf.output(dest='S')
    
    # Handle different FPDF versions
    if isinstance(pdf_output, str):
        # Older FPDF versions return string
        pdf_buffer.write(pdf_output.encode('latin-1'))
    else:
        # Newer FPDF versions return bytes
        pdf_buffer.write(pdf_output)
    
    # Get the bytes value
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
