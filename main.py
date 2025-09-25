import os
import re
from typing import List

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from fpdf import FPDF

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------- Streamlit Page Setup ----------------------
st.set_page_config(
    page_title="🎓 Smart Lecture Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .stApp {
        background-color: #F0F2F6;
    }
    .stButton>button {
        width: 100%;
    }
    .stTextInput>div>div>input {
        background-color: #FFFFFF;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------- Session State Initialization ----------------------
def init_session_state():
    """Initialize session state variables if they don't exist."""
    defaults = {
        "transcript_text": "",
        "summary_output": "",
        "chat_history": [],
        "show_study_plan": False,
        "study_plan": "",
        "last_video_url": "",
        "google_api_key": None,
        "llm": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ---------------------- Helpers ----------------------
YOUTUBE_ID_REGEX = re.compile(r"(?:v=|/)([0-9A-Za-z_-]{11}).*")

def extract_video_id(url: str) -> str:
    """Extract a YouTube video ID from common URL formats."""
    if not url:
        return ""
    if "youtu.be/" in url:
        return url.rstrip("/").split("/")[-1].split("?")[0]
    m = YOUTUBE_ID_REGEX.search(url)
    return m.group(1) if m else ""

@st.cache_data(show_spinner="Fetching transcript...")
def fetch_transcript(video_id: str, languages: List[str] = ["en", "en-US", "en-GB"]) -> str:
    """Fetch transcript text using youtube_transcript_api."""
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        text = " ".join(seg.get("text", "") for seg in segments if seg.get("text"))
        return re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        st.error(f"Transcript fetch error: Could not retrieve a transcript for the requested languages. Please check if the video has English captions. Details: {e}")
        return ""

def chunk_text(text: str, max_tokens: int = 2000) -> List[str]:
    """Simple char-based chunker as a proxy for token limits."""
    if not text:
        return []
    chunk_size = max_tokens * 4
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def run_llm(prompt: str) -> str:
    """Invoke the LLM and return its string content."""
    if not st.session_state.llm:
        st.error("LLM not initialized. Please configure your API key in the sidebar.")
        return ""
    try:
        resp = st.session_state.llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        return f"LLM Error: {e}"

def summarize_transcript(transcript: str) -> str:
    """Summarize the transcript into structured notes."""
    summary_template = PromptTemplate.from_template(
        "You are an expert note-taker. Summarize the following transcript chunk "
        "into structured notes with sections: Key Concepts, Definitions, Examples, "
        "Formulas, Action Items, and 5 Quiz Questions. Be concise and faithful to the content.\n\n"
        "Transcript chunk:\n{chunk}"
    )
    combine_template = PromptTemplate.from_template(
        "Combine these partial summaries into one cohesive summary with the same sections. "
        "Merge and deduplicate the quiz questions. Keep the total word count under 500.\n\n"
        "Partial summaries:\n{parts}"
    )

    chunks = chunk_text(transcript, max_tokens=2000)
    if not chunks:
        return "The transcript is empty, nothing to summarize."

    part_summaries = []
    progress_bar = st.progress(0, text="Summarizing chunks...")
    for i, ch in enumerate(chunks):
        prompt = summary_template.format(chunk=ch)
        part_summaries.append(run_llm(prompt))
        progress_bar.progress((i + 1) / len(chunks), text=f"Summarizing chunk {i+1}/{len(chunks)}")

    progress_bar.empty()
    if len(part_summaries) == 1:
        return part_summaries[0]

    with st.spinner("Combining summaries..."):
        combined_prompt = combine_template.format(parts="\n\n---\n\n".join(part_summaries))
        return run_llm(combined_prompt)

def answer_question(question: str, transcript: str, summary: str) -> str:
    """Answer a question grounded in the transcript and summary."""
    qa_template = PromptTemplate.from_template(
        "You are a helpful teaching assistant. Answer the question using ONLY information "
        "from the lecture summary and transcript. If the answer is not present, reply: "
        "'This information is not covered in the lecture.' Keep answers concise.\n\n"
        "Summary:\n{summary}\n\nTranscript:\n{transcript}\n\nQuestion: {question}"
    )
    prompt = qa_template.format(
        question=question,
        summary=summary,
        transcript=transcript[:15000],
    )
    return run_llm(prompt)

@st.cache_data(show_spinner="Building study plan...")
def build_study_plan(summary: str) -> str:
    """Create a 7-day study plan from the summary."""
    plan_template = PromptTemplate.from_template(
        "Create a concise 7-day study plan from the lecture summary. For each day, "
        "include: goals, study tasks with time estimates, a 3-question self-check, "
        "and a checkpoint outcome.\n\nSummary:\n{summary}"
    )
    return run_llm(plan_template.format(summary=summary))

def pdf_bytes_from_text(title: str, content: str) -> bytes:
    """Render text into a PDF and return raw bytes."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.multi_cell(0, 10, txt=title)
    pdf.ln(5)
    pdf.set_font("Arial", size=11)
    safe_content = content.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 7, txt=safe_content)
    return bytes(pdf.output(dest="S").encode("latin-1"))

# ---------------------- Sidebar UI ----------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.session_state.google_api_key = st.text_input(
        "Enter your Google Gemini API Key", type="password"
    )

    model_choice = st.selectbox(
        "Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro"],
        index=0, help="Flash for speed, Pro for higher quality."
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05,
        help="Controls the creativity of the model. Lower is more deterministic.")

    if st.button("Initialize Model"):
        if st.session_state.google_api_key:
            try:
                st.session_state.llm = ChatGoogleGenerativeAI(
                    model=model_choice,
                    google_api_key=st.session_state.google_api_key,
                    temperature=temperature,
                )
                st.success("Gemini model initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize model: {e}")
        else:
            st.warning("Please enter your Gemini API key.")

    st.markdown("---")
    st.info("This app uses Google's Gemini to summarize YouTube lecture videos and answer your questions about them.")


# ---------------------- Main UI ----------------------
st.title("🎓 Smart Lecture Summarizer")
st.markdown("Your intelligent assistant for absorbing video lectures. Just paste a YouTube URL to get started.")

if not st.session_state.llm:
    st.warning("Please configure your API key and initialize the model in the sidebar to proceed.", icon="👈")
else:
    st.subheader("1. Enter YouTube Video URL")
    video_url = st.text_input(
        "YouTube URL",
        placeholder="e.g., https://www.youtube.com/watch?v=...",
        value=st.session_state.last_video_url,
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        summarize_btn = st.button("✨ Generate Summary", type="primary")
    with col2:
        clear_btn = st.button("🗑️ Clear All")

    if clear_btn:
        for key in st.session_state.keys():
            if key not in ['google_api_key', 'llm']: # Keep API key and model
                 st.session_state[key] = "" if isinstance(st.session_state[key], str) else [] if isinstance(st.session_state[key], list) else False
        st.rerun()

    if summarize_btn:
        vid = extract_video_id(video_url)
        if not vid:
            st.error("Invalid YouTube URL. Please enter a valid URL.")
        else:
            st.session_state.last_video_url = video_url
            transcript = fetch_transcript(vid)
            if transcript:
                st.session_state.transcript_text = transcript
                st.session_state.summary_output = summarize_transcript(transcript)
                st.success("Summary generated successfully!")
                st.session_state.chat_history = []
                st.session_state.show_study_plan = False

# Display results
if st.session_state.summary_output:
    st.markdown("---")
    st.subheader("📝 Lecture Summary")
    st.markdown(st.session_state.summary_output)

    with st.expander("View Full Transcript"):
        st.text_area("Transcript", st.session_state.transcript_text, height=200)

    # Action buttons
    c1, c2 = st.columns(2)
    with c1:
        summary_pdf_bytes = pdf_bytes_from_text("Lecture Summary", st.session_state.summary_output)
        st.download_button(
            "⬇️ Download Summary PDF",
            data=summary_pdf_bytes,
            file_name="lecture_summary.pdf",
            mime="application/pdf",
        )
    with c2:
        if st.button("🗓️ Create 7-Day Study Plan"):
            st.session_state.study_plan = build_study_plan(st.session_state.summary_output)
            st.session_state.show_study_plan = True

if st.session_state.get("show_study_plan"):
    st.subheader("📚 Your 7-Day Study Plan")
    st.markdown(st.session_state.study_plan)
    plan_pdf_bytes = pdf_bytes_from_text("7-Day Study Plan", st.session_state.study_plan)
    st.download_button(
        "⬇️ Download Plan PDF",
        data=plan_pdf_bytes,
        file_name="study_plan.pdf",
        mime="application/pdf",
    )

# Q&A Section
if st.session_state.summary_output:
    st.markdown("---")
    st.subheader("💬 Ask Questions About the Lecture")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if user_q := st.chat_input("What would you like to know?"):
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ans = answer_question(
                    user_q,
                    transcript=st.session_state.transcript_text,
                    summary=st.session_state.summary_output,
                )
                st.markdown(ans)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})

