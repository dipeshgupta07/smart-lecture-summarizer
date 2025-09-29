import streamlit as st
from dotenv import load_dotenv
import os
import re
from fpdf import FPDF
from transformers import pipeline
import PyPDF2
import io # Added to support in-memory PDF generation

load_dotenv()

# ---------------------- Streamlit Page Setup ----------------------
st.set_page_config(page_title="📄 Smart PDF Summarizer", page_icon="📚")
st.title("📄 Smart PDF Summarizer")

# ---------------------- Initialize session state ----------------------
for key, default in {
    "pdf_text": "",
    "summary_output": "",
    "last_pdf_name": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------- Functions ----------------------
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    try:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)
                # Extract text from all pages
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF. The PDF might contain only images or be password protected.")
                return text.strip()
    except Exception as e:
        raise ValueError(f"Error reading PDF: {str(e)}")

def remove_emojis(text):
    """Remove emojis and non-ASCII characters from text."""
    # This regex matches any character that is NOT an ASCII character
    return re.sub(r'[^\x00-\x7F]+', '', text)

def clean_text(text):
    """Clean extracted PDF text by removing extra whitespace and fixing formatting."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers and headers/footers (basic cleanup)
    text = re.sub(r'\n\d+\n', '\n', text)
    # Fix common PDF extraction issues like "WordJoins" -> "Word Joins"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text.strip()

# ====================================================================
# *** CORRECTED PDF GENERATION FUNCTION ***
# ====================================================================
def generate_pdf(text, filename="Document_Summary.pdf"):
    """
    Generate a PDF file from text using FPDF, handling encoding issues 
    and returning the content as bytes using io.BytesIO.
    """
    from fpdf import FPDF
    
    pdf = FPDF()
    pdf.add_page()
    
    # Use a standard font like Arial
    pdf.set_font("Arial", size=10)

    # Split text into lines
    lines = text.split("\n")
    
    for line in lines:
        if line.strip():
            # Crucial Fix: Encode the line using 'latin-1' (which FPDF/Arial supports)
            # and replace any unsupported Unicode characters with '?'
            clean_line = line.encode('latin-1', 'replace').decode('latin-1')
            
            # multi_cell automatically wraps text.
            # 0 for width means 'as wide as possible', 6 for line height.
            pdf.multi_cell(0, 6, clean_line)
        else:
            # Add spacing for empty lines
            pdf.ln(3)
            
    # Save the PDF to the in-memory buffer (dest='S' returns the document as a string)
    # We explicitly encode it to bytes to pass to st.download_button
    pdf_content = pdf.output(dest='S').encode('latin-1')
    
    # Return the bytes data directly
    return pdf_content
# ====================================================================


# ---------------------- Model Setup ----------------------
@st.cache_resource
def load_models():
    """Load and cache the AI models."""
    try:
        # Use a proper summarization model
        # Using a fallback is good practice
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        return summarizer
    except Exception as e:
        st.error(f"Error loading models: {e}. Falling back to a smaller model...")
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        return summarizer

summarizer = load_models()

def extract_key_sentences(text, num_sentences=5):
    """Extract key sentences using simple heuristics."""
    sentences = text.split('. ')
        # Filter out very short sentences
    sentences = [s.strip() + '.' for s in sentences if len(s.strip()) > 20]
        # Score sentences based on length and position
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        if i >= 20: # Only consider first 20 for scoring efficiency
            break
        # Give higher scores to sentences in the beginning and middle
        position_score = 1.0 if i < 3 else 0.7 if i < 10 else 0.5
        length_score = min(len(sentence) / 100, 1.0)  # Prefer medium-length sentences
        total_score = position_score * length_score
        scored_sentences.append((sentence, total_score))
        # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored_sentences[:num_sentences]]

def create_smart_notes(text):
    """Create structured notes from text using keyword extraction."""
    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        if not paragraphs:
        # Fallback if standard paragraph splitting fails
        paragraphs = [text[i:i+1000] for i in range(0, min(len(text), 3000), 1000)]
        paragraphs = [p for p in paragraphs if p.strip()]
        
    notes = []
    for i, paragraph in enumerate(paragraphs[:3]):  # Limit to 3 sections
        # Extract first sentence as heading
        sentences = paragraph.split('. ')
        if sentences:
            # Clean heading: limit length and ensure it ends with a proper mark if trimmed
            heading = sentences[0]
            if len(heading) > 80:
                 heading = heading[:80].rsplit(' ', 1)[0] + '...' # Trim gracefully
            if not heading.endswith('.'):
                 heading += '.'

            # Create bullet points from remaining sentences
            bullets = []
            for sentence in sentences[1:4]:  # Max 3 bullets per section
                if len(sentence.strip()) > 10:
                    bullet = sentence.strip()
                    if not bullet.endswith('.'):
                        bullet += '.'
                    bullets.append(f"- {bullet}")
                        if bullets:
                notes.append(f"**{heading}**\n" + "\n".join(bullets))

    return notes


def generate_summary(pdf_text):
    """Generate a summary from PDF text using proper summarization."""
    try:
        # Clean and prepare text
        text = pdf_text.strip()
        if len(text) < 100:
            return "Document too short to summarize effectively."

        # Use AI summarization for main summary
        summary_text = ""
        
        # Transformer model limit is typically 1024 tokens (about 3000-4000 characters)
        if len(text) > 3000:
            # Limit text to the first 3000 characters for initial summarization chunk
            text_chunk = text[:3000]
        else:
            text_chunk = text

        # Summarize the chunk
        if len(text_chunk.strip()) > 100:
            try:
                # Use model summarization
                result = summarizer(text_chunk, max_length=150, min_length=50, do_sample=False)
                summary_text = result[0]['summary_text']
            except Exception as e:
                # Fallback to extractive if model fails
                print(f"Model summarization failed: {e}")
                key_sentences = extract_key_sentences(text_chunk, 5)
                summary_text = " ".join(key_sentences)
        
        # Extract key points for bullet format (from the whole text if possible, or chunk)
        key_sentences = extract_key_sentences(text, 5)
                
        # Create smart notes
        smart_notes = create_smart_notes(text)
                
        # Format the final output
        formatted_summary = "📋 **Summary:**\n"
        if summary_text:
            formatted_summary += f"{summary_text}\n\n"
        
        formatted_summary += "🎯 **Key Takeaways:**\n"
        # Add key points
        for sentence in key_sentences[:4]:
            formatted_summary += f"- {sentence}\n"
                
        # Add smart notes section
        if smart_notes:
            formatted_summary += "\n📝 **Smart Notes/Structure:**\n"
            formatted_summary += "\n---\n".join(smart_notes)
            
        return formatted_summary
            
    except Exception as e:
        # Complete fallback to extractive summarization
        print(f"Major summarization error: {e}")
        key_sentences = extract_key_sentences(pdf_text, 6)
        fallback_summary = "❌ **Error: Could not generate AI Summary.**\n\n🎯 **Extractive Fallback Key Sentences:**\n" + '\n'.join([f"- {s}" for s in key_sentences])
        return fallback_summary


# ---------------------- Streamlit UI ----------------------
st.markdown("Upload a PDF document to get an AI-generated summary with smart notes!")

# File uploader
uploaded_file = st.file_uploader(
    "📎 Choose a PDF file", 
    type="pdf",
    help="Upload a PDF document to summarize")

if uploaded_file is not None:
    # Check if this is a new file
    if uploaded_file.name != st.session_state.last_pdf_name:
        st.session_state.last_pdf_name = uploaded_file.name
        st.session_state.summary_output = ""
                
        with st.spinner("📖 Extracting text from PDF..."):
            try:
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                # Extract text from PDF
                raw_text = extract_text_from_pdf(uploaded_file)
                # Clean the extracted text
                st.session_state.pdf_text = clean_text(raw_text)
                st.success("✅ Text extracted successfully!")
                                
                # Show file info
                st.info(f"📄 File: {uploaded_file.name} | Characters extracted: {len(st.session_state.pdf_text):,}")
                            
            except Exception as e:
                st.error(f"❌ Error extracting text from PDF: {e}")
                st.stop()
    
    # Show text preview if available
    if st.session_state.pdf_text:
        with st.expander("📖 Text Preview"):
            preview_text = st.session_state.pdf_text[:2000]
            if len(st.session_state.pdf_text) > 2000:
                preview_text += "\n\n... (Text truncated for preview) ..."
            st.text_area(
                "Extracted Text Preview:",
                preview_text,
                height=300,
                disabled=True
            )

        # Summarize button
        if st.button("🧠 Summarize Document"):
            with st.spinner("⏳ Generating summary and smart notes..."):
                try:
                    summary = generate_summary(st.session_state.pdf_text)
                    # Clean emojis before saving to session state to prevent download issues
                    clean_summary = remove_emojis(summary) 
                    st.session_state.summary_output = clean_summary
                    st.success("✅ Summary and Notes Generated!")
                except Exception as e:
                    st.error(f"❌ Summarization Error: {e}")

        # Display summary if available
        if st.session_state.summary_output:
            st.markdown("### 📋 Summary & 📝 Smart Notes")
            # Use st.markdown for proper formatting (bold, lists, etc.)
            st.markdown(st.session_state.summary_output)
            
            # Download PDF button
            try:
                # CALL THE NEW FUNCTION - IT RETURNS BYTES
                pdf_bytes = generate_pdf(st.session_state.summary_output)
                
                # Use the returned bytes data for the download button
                st.download_button(
                    label="💾 Download Summary as PDF",
                    data=pdf_bytes,
                    file_name="Document_Summary.pdf",
                    mime="application/pdf",
                )
                
            except Exception as e:
                # This catches any residual FPDF-related errors
                st.warning(f"PDF generation failed: {e}")
                
else:
    st.info("👆 Please upload a PDF file to get started!")

# Footer
st.markdown("---")
st.markdown("🔧 **Tips for better results:**")
st.markdown("- Upload PDFs with clear, readable text")
st.markdown("- Avoid heavily formatted documents or scanned images")
st.markdown("- For large documents, the summarizer focuses on the first ~3000 characters")
