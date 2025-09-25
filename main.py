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
    try:
        # Use a proper summarization model
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        # Keep a smaller text generation model for Q&A
        text_generator = pipeline("text-generation", model="distilgpt2")
        return summarizer, text_generator
    except Exception as e:
        st.error(f"Error loading models: {e}")
        # Fallback to a smaller summarization model
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        text_generator = pipeline("text-generation", model="distilgpt2")
        return summarizer, text_generator

summarizer, text_generator = load_models()


def extract_key_sentences(text, num_sentences=5):
    """Extract key sentences using simple heuristics."""
    sentences = text.split('. ')
    
    # Filter out very short sentences
    sentences = [s.strip() + '.' for s in sentences if len(s.strip()) > 20]
    
    # Score sentences based on length and position
    scored_sentences = []
    for i, sentence in enumerate(sentences[:20]):  # Only consider first 20 sentences
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
        paragraphs = [text[:500], text[500:1000], text[1000:1500]]
        paragraphs = [p for p in paragraphs if p.strip()]
    
    notes = []
    for i, paragraph in enumerate(paragraphs[:3]):  # Limit to 3 sections
        # Extract first sentence as heading
        sentences = paragraph.split('. ')
        if sentences:
            heading = sentences[0][:60] + "..." if len(sentences[0]) > 60 else sentences[0]
            # Create bullet points from remaining sentences
            bullets = []
            for sentence in sentences[1:4]:  # Max 3 bullets per section
                if len(sentence.strip()) > 10:
                    bullet = sentence.strip()
                    if not bullet.endswith('.'):
                        bullet += '.'
                    bullets.append(f"- {bullet}")
            
            if bullets:
                notes.append(f"{heading}:\n" + "\n".join(bullets))
    
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
        if len(text) > 1024:  # BART has token limits
            # Split text into chunks
            chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
            chunk_summaries = []
            
            for chunk in chunks[:3]:  # Limit to first 3 chunks
                if len(chunk.strip()) > 100:
                    try:
                        result = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
                        chunk_summaries.append(result[0]['summary_text'])
                    except Exception:
                        # Fallback to extractive for this chunk
                        key_sentences = extract_key_sentences(chunk, 2)
                        chunk_summaries.extend(key_sentences)
            
            summary_text = " ".join(chunk_summaries)
        else:
            # Summarize the whole text
            try:
                result = summarizer(text, max_length=150, min_length=50, do_sample=False)
                summary_text = result[0]['summary_text']
            except Exception:
                # Fallback to extractive summarization
                key_sentences = extract_key_sentences(text, 5)
                summary_text = " ".join(key_sentences)
        
        # Extract key points for bullet format
        key_sentences = extract_key_sentences(text, 5)
        
        # Create smart notes
        smart_notes = create_smart_notes(text)
        
        # Format the final output
        formatted_summary = "📋 Summary:\n"
        if summary_text:
            formatted_summary += f"- {summary_text}\n\n"
        
        # Add key points
        for sentence in key_sentences[:4]:
            formatted_summary += f"- {sentence}\n"
        
        # Add smart notes section
        if smart_notes:
            formatted_summary += "\n📝 Smart Notes:\n"
            for note in smart_notes:
                formatted_summary += f"\n{note}\n"
        
        return formatted_summary
        
    except Exception as e:
        # Complete fallback to extractive summarization
        key_sentences = extract_key_sentences(pdf_text, 6)
        fallback_summary = "📋 Summary:\n" + '\n'.join([f"- {s}" for s in key_sentences])
        return fallback_summary


def generate_answer(summary, chat_history, new_question):
    """Generate an answer based on the summary and chat history."""
    # Keep context short and relevant
    context = summary[:800] if len(summary) > 800 else summary
    
    # Simple prompt for better results
    prompt = f"Context: {context}\n\nQuestion: {new_question}\nAnswer:"
    
    try:
        results = text_generator(
            prompt, 
            max_length=len(prompt) + 80,
            num_return_sequences=1, 
            do_sample=True, 
            temperature=0.7,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            pad_token_id=text_generator.tokenizer.eos_token_id
        )
        generated_text = results[0]["generated_text"]
        answer = generated_text[len(prompt):].strip()
        
        # Clean up the answer and limit length
        if answer:
            # Take only the first sentence or two
            sentences = answer.split('. ')
            answer = '. '.join(sentences[:2])
            if not answer.endswith('.'):
                answer += '.'
            return answer
        else:
            return "I couldn't find relevant information in the document to answer this question."
            
    except Exception as e:
        return "Sorry, I encountered an error while generating the answer. Please try rephrasing your question."


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
