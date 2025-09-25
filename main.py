import streamlit as st
from dotenv import load_dotenv
import os
import re
from fpdf import FPDF
from transformers import pipeline
import PyPDF2
import io

load_dotenv()

# ---------------------- Streamlit Page Setup ----------------------
st.set_page_config(page_title="📄 Smart PDF Summarizer", page_icon="📚")
st.title("📄 Smart PDF Summarizer")

# ---------------------- Initialize session state ----------------------
for key, default in {
    "pdf_text": "",
    "summary_output": "",
    "chat_history": [],
    "show_study_plan": False,
    "question_submitted": False,
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
    return re.sub(r"[^\x00-\x7F]+", "", text)


def clean_text(text):
    """Clean extracted PDF text by removing extra whitespace and fixing formatting."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers and headers/footers (basic cleanup)
    text = re.sub(r'\n\d+\n', '\n', text)
    # Fix common PDF extraction issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text.strip()


def generate_pdf(text, filename="summary_notes.pdf"):
    """Generate a PDF file from text."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Calculate the available width for the cell
    cell_width = pdf.w - 2 * pdf.l_margin
    if cell_width <= 0:
        raise Exception("Calculated PDF cell width is zero or negative. Check page dimensions or margins.")
    
    # Split text into lines and add to PDF
    for line in text.split("\n"):
        if line.strip():  # Only add non-empty lines
            pdf.multi_cell(cell_width, 10, line.encode('latin-1', 'replace').decode('latin-1'))
    
    pdf.output(filename)
    return filename


def build_chat_prompt(summary, chat_history, new_question):
    """Build a prompt for the chat functionality."""
    prompt = f"""You are a helpful assistant answering questions based on the following document summary:

{summary}

"""
    if chat_history:
        prompt += "Here is the conversation so far:\n"
        for q, a in chat_history:
            prompt += f"Q: {q}\nA: {a}\n"
    prompt += f"\nQ: {new_question}\n\nAnswer:"
    return prompt


# ---------------------- Model Setup ----------------------
@st.cache_resource
def load_models():
    """Load and cache the AI models."""
    text_generator = pipeline("text-generation", model="distilgpt2")
    return text_generator

text_generator = load_models()


def generate_summary(pdf_text):
    """Generate a summary from PDF text."""
    # Truncate text if too long (DistilGPT2 has token limits)
    max_chars = 3000
    if len(pdf_text) > max_chars:
        pdf_text = pdf_text[:max_chars] + "..."
    
    prompt = f"""
You are a helpful study assistant.

Given the following document text:

{pdf_text}

Please:
1. Summarize it in bullet points.
2. Suggest smart, self-explanatory note headings and bullet points.

Format exactly as:

📋 Summary:
- Main point 1
- Main point 2
- Main point 3

📝 Smart Notes:
Key Topic 1:
- Important detail A
- Important detail B

Key Topic 2:
- Important detail C
- Important detail D
"""
    
    try:
        results = text_generator(prompt, max_length=len(prompt) + 300, num_return_sequences=1, do_sample=True, temperature=0.7)
        generated_text = results[0]["generated_text"]
        # Extract only the generated part (after the prompt)
        summary = generated_text[len(prompt):].strip()
        return summary if summary else "Summary could not be generated. Please try with a different document."
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def generate_answer(summary, chat_history, new_question):
    """Generate an answer based on the summary and chat history."""
    prompt = build_chat_prompt(summary, chat_history, new_question)
    
    try:
        results = text_generator(prompt, max_length=len(prompt) + 150, num_return_sequences=1, do_sample=True, temperature=0.7)
        generated_text = results[0]["generated_text"]
        answer = generated_text[len(prompt):].strip()
        return answer if answer else "I couldn't generate a proper answer. Please try rephrasing your question."
    except Exception as e:
        return f"Error generating answer: {str(e)}"


# ---------------------- Streamlit UI ----------------------
st.markdown("Upload a PDF document to get an AI-generated summary and ask questions about it!")

# File uploader
uploaded_file = st.file_uploader(
    "📎 Choose a PDF file", 
    type="pdf",
    help="Upload a PDF document to summarize"
)

if uploaded_file is not None:
    # Check if this is a new file
    if uploaded_file.name != st.session_state.last_pdf_name:
        st.session_state.last_pdf_name = uploaded_file.name
        st.session_state.summary_output = ""
        st.session_state.chat_history = []
        st.session_state.show_study_plan = False
        st.session_state.question_submitted = False
        
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
                preview_text += "..."
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
                    clean_summary = remove_emojis(summary)
                    st.session_state.summary_output = clean_summary
                    st.session_state.chat_history = []
                    st.session_state.show_study_plan = False
                    st.session_state.question_submitted = False
                    st.success("✅ Summary and Notes Generated!")
                except Exception as e:
                    st.error(f"❌ Summarization Error: {e}")

        # Display summary if available
        if st.session_state.summary_output:
            st.markdown("### 📋 Summary & 📝 Smart Notes")
            st.markdown(st.session_state.summary_output)

            # Download PDF button
            try:
                pdf_file = generate_pdf(st.session_state.summary_output)
                with open(pdf_file, "rb") as file:
                    st.download_button(
                        label="💾 Download Summary as PDF",
                        data=file,
                        file_name="Document_Summary.pdf",
                        mime="application/pdf",
                    )
            except Exception as e:
                st.warning(f"PDF generation failed: {e}")

            # Chat interface
            st.divider()
            st.subheader("💬 Ask questions about the document")

            with st.form(key="chat_form", clear_on_submit=True):
                question = st.text_input(
                    "Ask a question about the document:", 
                    key="chat_input",
                    placeholder="e.g., What are the main conclusions?"
                )
                submit_button = st.form_submit_button("❓ Get Answer")

                if submit_button:
                    if question.strip() == "":
                        st.warning("Please enter a question!")
                    else:
                        with st.spinner("🤖 Thinking..."):
                            try:
                                answer = generate_answer(
                                    st.session_state.summary_output,
                                    st.session_state.chat_history,
                                    question,
                                )
                                st.session_state.chat_history.append((question, answer))
                                st.session_state.question_submitted = True
                            except Exception as e:
                                st.error(f"❌ Error getting answer: {e}")

            # Display chat history
            if st.session_state.chat_history:
                st.markdown("### 💭 Conversation History")
                for q, a in reversed(st.session_state.chat_history):
                    st.markdown(f"**Q:** {q}")
                    st.markdown(f"**A:** {a}")
                    st.markdown("---")

else:
    st.info("👆 Please upload a PDF file to get started!")

# Footer
st.markdown("---")
st.markdown("🔧 **Tips for better results:**")
st.markdown("- Upload PDFs with clear, readable text")
st.markdown("- Avoid heavily formatted documents or scanned images")
st.markdown("- For large documents, the summarizer works on the first ~3000 characters")
st.markdown("- Ask specific questions about the content for better answers")
