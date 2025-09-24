import os
import re
from typing import List
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi

# Try to import reportlab, fallback if not available
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from io import BytesIO
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("⚠️ PDF export not available. Install reportlab for PDF functionality.")

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------- Configuration ----------------------
st.set_page_config(
    page_title="🎓 Smart Lecture Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .step-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .feature-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .chat-container {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .summary-container {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------- Initialize Session State ----------------------
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = ""
if "summary_output" not in st.session_state:
    st.session_state.summary_output = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_study_plan" not in st.session_state:
    st.session_state.show_study_plan = False
if "study_plan" not in st.session_state:
    st.session_state.study_plan = ""
if "last_video_url" not in st.session_state:
    st.session_state.last_video_url = ""
if "video_title" not in st.session_state:
    st.session_state.video_title = ""
if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = ""

# ---------------------- Helper Functions ----------------------
YOUTUBE_ID_REGEX = re.compile(r"(?:v=|/)([0-9A-Za-z_-]{11}).*")

def extract_video_id(url: str) -> str:
    """Extract a YouTube video ID from common URL formats."""
    if not url:
        return ""
    if "youtu.be/" in url:
        return url.rstrip("/").split("/")[-1].split("?")[0]
    m = YOUTUBE_ID_REGEX.search(url)
    return m.group(1) if m else ""

def fetch_transcript(video_id: str, languages: List[str] = ["en", "en-US", "en-GB"]) -> str:
    """Fetch transcript text using youtube_transcript_api."""
    segments = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    text = " ".join(seg.get("text", "") for seg in segments if seg.get("text"))
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text: str, max_tokens: int = 2000) -> List[str]:
    """Simple char-based chunker as a proxy for token limits."""
    if not text:
        return []
    chunk_size = max_tokens * 4  # rough char-token heuristic
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def run_llm(prompt: str, llm) -> str:
    """Invoke Gemini through LangChain chat model and return string content."""
    try:
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        return f"LLM Error: {e}"

def summarize_transcript(transcript: str, llm) -> str:
    """Summarize the transcript into structured notes."""
    summary_template = PromptTemplate(
        input_variables=["chunk"],
        template=(
            "You are an expert note-taker for university lectures.\n"
            "Summarize the following transcript chunk into structured notes with these sections:\n"
            "- **Key Concepts**\n"
            "- **Definitions**\n"
            "- **Examples**\n"
            "- **Important Formulas** (use plain text)\n"
            "- **Action Items**\n"
            "- **5 Quiz Questions**\n\n"
            "Be concise and faithful to the content. Avoid speculation.\n\n"
            "Transcript chunk:\n{chunk}"
        ),
    )

    combine_template = PromptTemplate(
        input_variables=["parts"],
        template=(
            "Combine the following partial summaries into one cohesive summary with the same sections:\n"
            "- **Key Concepts**\n"
            "- **Definitions**\n"
            "- **Examples**\n"
            "- **Important Formulas**\n"
            "- **Action Items**\n"
            "- **10 Quiz Questions** (merge and deduplicate)\n\n"
            "Ensure clarity, remove duplication, and keep it comprehensive but organized.\n\n"
            "Partial summaries:\n{parts}"
        ),
    )

    chunks = chunk_text(transcript, max_tokens=2000)
    part_summaries = []
    
    progress_bar = st.progress(0)
    for idx, ch in enumerate(chunks, 1):
        progress = idx / len(chunks)
        progress_bar.progress(progress, text=f"Summarizing chunk {idx}/{len(chunks)}...")
        prompt = summary_template.format(chunk=ch)
        part_summaries.append(run_llm(prompt, llm))

    progress_bar.empty()
    
    if len(part_summaries) == 1:
        return part_summaries[0]

    combined = combine_template.format(parts="\n\n---\n\n".join(part_summaries))
    return run_llm(combined, llm)

def answer_question(question: str, transcript: str, summary: str, llm) -> str:
    """Answer a question grounded strictly in the transcript and summary."""
    qa_template = PromptTemplate(
        input_variables=["question", "summary", "transcript"],
        template=(
            "You are a helpful teaching assistant.\n"
            "Answer the question using ONLY the information from the lecture summary and transcript.\n"
            "If something is not covered, reply: \"Not covered in this transcript.\" Keep answers under 150 words.\n\n"
            "Summary:\n{summary}\n\n"
            "Transcript:\n{transcript}\n\n"
            "Question: {question}"
        ),
    )
    prompt = qa_template.format(
        question=question,
        summary=summary,
        transcript=transcript[:15000],  # safety truncation
    )
    return run_llm(prompt, llm)

def build_study_plan(summary: str, llm) -> str:
    """Create a concise 7-day study plan based on the summary."""
    plan_template = PromptTemplate(
        input_variables=["summary"],
        template=(
            "Create a detailed 7-day study plan based on the lecture summary below. "
            "Each day should include:\n"
            "- **Day X Goals**\n"
            "- **Study Tasks** with time estimates\n"
            "- **3 Self-Check Questions**\n"
            "- **Checkpoint Outcome**\n\n"
            "Make it practical and achievable.\n\n"
            "Summary:\n{summary}"
        ),
    )
    return run_llm(plan_template.format(summary=summary), llm)

def pdf_bytes_from_text(title: str, content: str) -> bytes:
    """Render simple text into a PDF and return raw bytes using reportlab."""
    if not PDF_AVAILABLE:
        return b"PDF generation not available"
    
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            textColor='#333333'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            textColor='#444444'
        )
        
        # Create story (content)
        story = []
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 12))
        
        # Split content into paragraphs
        paragraphs = content.split('\n')
        for para in paragraphs:
            if para.strip():
                # Escape HTML characters and handle special characters
                safe_para = para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(safe_para, body_style))
                story.append(Spacer(1, 6))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return b"PDF generation failed"

# ---------------------- Sidebar Configuration ----------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # API Key Input
    api_key_input = st.text_input(
        "Google API Key",
        value=st.session_state.GOOGLE_API_KEY,
        type="password",
        help="Enter your Google Gemini API Key"
    )
    
    if api_key_input != st.session_state.GOOGLE_API_KEY:
        st.session_state.GOOGLE_API_KEY = api_key_input
    
    if not st.session_state.GOOGLE_API_KEY:
        st.error("⚠️ Please enter your Google API Key to continue")
        st.info("Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)")
    
    st.divider()
    
    # Model Configuration
    model_choice = st.selectbox(
        "🤖 Gemini Model",
        options=["gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
        help="Flash: Faster, Pro: Higher quality"
    )
    
    temperature = st.slider(
        "🌡️ Temperature", 
        0.0, 1.0, 0.2, 0.05,
        help="Controls creativity: 0=focused, 1=creative"
    )
    
    st.divider()
    
    # Features Info
    st.markdown("### 🚀 Features")
    st.markdown("""
    - **Auto Transcript** extraction
    - **Smart Summarization** with structure
    - **Q&A Chat** based on content
    - **Study Plans** generation
    - **Text Export** functionality
    """)
    
    if PDF_AVAILABLE:
        st.success("✅ PDF Export Available")
    else:
        st.warning("⚠️ PDF Export Disabled")
    
    if st.button("🗑️ Clear All Data", type="secondary"):
        for key in ['transcript_text', 'summary_output', 'chat_history', 'show_study_plan', 'study_plan', 'last_video_url', 'video_title']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ---------------------- Main Application ----------------------

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 Smart Lecture Summarizer</h1>
    <p>Transform YouTube lectures into structured notes, study plans, and interactive Q&A</p>
</div>
""", unsafe_allow_html=True)

# Check if API key is available
if not st.session_state.GOOGLE_API_KEY:
    st.error("🔑 Please enter your Google API Key in the sidebar to continue")
    st.stop()

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(
        model=model_choice,
        google_api_key=st.session_state.GOOGLE_API_KEY,
        temperature=temperature,
    )
except Exception as e:
    st.error(f"❌ Error initializing LLM: {e}")
    st.stop()

# Step 1: URL Input
st.markdown('<div class="step-card">', unsafe_allow_html=True)
st.markdown("### 📎 Step 1: Enter YouTube URL")

col1, col2 = st.columns([3, 1])
with col1:
    video_url = st.text_input(
        "",
        placeholder="https://www.youtube.com/watch?v=VIDEO_ID or https://youtu.be/VIDEO_ID",
        value=st.session_state.last_video_url,
        label_visibility="collapsed"
    )

with col2:
    summarize_btn = st.button("🎯 Analyze Video", type="primary")

st.markdown('</div>', unsafe_allow_html=True)

# Process Video
if summarize_btn and video_url:
    vid = extract_video_id(video_url)
    if not vid:
        st.markdown('<div class="error-box">❌ Could not extract a valid YouTube video ID from the URL.</div>', unsafe_allow_html=True)
    else:
        try:
            with st.spinner("🔍 Fetching transcript..."):
                transcript = fetch_transcript(vid, languages=["en", "en-US", "en-GB"])
            
            st.session_state.transcript_text = transcript
            st.session_state.last_video_url = video_url
            st.session_state.video_title = f"YouTube Video: {vid}"
            
            with st.spinner("🧠 Generating summary with AI..."):
                st.session_state.summary_output = summarize_transcript(transcript, llm)
            
            st.markdown('<div class="success-box">✅ Summary generated successfully!</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)

# Step 2: Display Summary
if st.session_state.summary_output:
    st.markdown("### 📝 Step 2: Lecture Summary")
    
    st.markdown('<div class="summary-container">', unsafe_allow_html=True)
    st.markdown(st.session_state.summary_output)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Text download as fallback
        st.download_button(
            "📄 Download Summary (TXT)",
            data=st.session_state.summary_output,
            file_name="lecture_summary.txt",
            mime="text/plain",
        )
        
        # PDF download if available
        if PDF_AVAILABLE:
            try:
                sum_pdf = pdf_bytes_from_text("Lecture Summary", st.session_state.summary_output)
                st.download_button(
                    "📄 Download Summary (PDF)",
                    data=sum_pdf,
                    file_name="lecture_summary.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
    
    with col2:
        if st.button("📅 Create Study Plan"):
            with st.spinner("🎯 Building personalized study plan..."):
                plan = build_study_plan(st.session_state.summary_output, llm)
                st.session_state.study_plan = plan
                st.session_state.show_study_plan = True
    
    with col3:
        if st.session_state.transcript_text:
            st.success(f"✅ {len(st.session_state.transcript_text.split())} words extracted")

# Step 3: Study Plan
if st.session_state.get("show_study_plan", False) and st.session_state.get("study_plan"):
    st.markdown("### 📚 Step 3: Your Personalized Study Plan")
    
    st.markdown('<div class="summary-container">', unsafe_allow_html=True)
    st.markdown(st.session_state.study_plan)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Text download
    st.download_button(
        "📄 Download Study Plan (TXT)",
        data=st.session_state.study_plan,
        file_name="study_plan.txt",
        mime="text/plain",
    )
    
    # PDF download if available
    if PDF_AVAILABLE:
        try:
            plan_pdf = pdf_bytes_from_text("7-Day Study Plan", st.session_state.study_plan)
            st.download_button(
                "📄 Download Study Plan (PDF)",
                data=plan_pdf,
                file_name="study_plan.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")

# Step 4: Interactive Q&A
st.markdown("### 💬 Step 4: Ask Questions About the Lecture")

if not st.session_state.transcript_text:
    st.info("👆 Please analyze a YouTube video first to enable the Q&A feature")
else:
    # Chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                st.markdown(f"**🙋 You:** {msg['content']}")
            else:
                st.markdown(f"**🤖 Assistant:** {msg['content']}")
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("---")
    else:
        st.markdown("*Start a conversation by asking a question about the lecture below...*")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    user_question = st.chat_input("💭 Ask about the lecture content...")
    
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        with st.spinner("🤔 Analyzing your question..."):
            answer = answer_question(
                user_question,
                transcript=st.session_state.transcript_text,
                summary=st.session_state.summary_output,
                llm=llm
            )
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🎓 Smart Lecture Summarizer | Powered by Google Gemini AI</p>
    <p><em>Transform your learning experience with AI-powered lecture analysis</em></p>
</div>
""", unsafe_allow_html=True)
