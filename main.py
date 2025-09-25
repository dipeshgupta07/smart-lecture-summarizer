import streamlit as st
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)
from dotenv import load_dotenv
import os
import re
from fpdf import FPDF
from transformers import pipeline

load_dotenv()

# ---------------------- Streamlit Page Setup ----------------------
st.set_page_config(page_title="🎓 Smart Lecture Summarizer", page_icon="📚")
st.title("🎓 Smart Lecture Summarizer")

# ---------------------- Initialize session state ----------------------
for key, default in {
    "transcript_text": "",
    "summary_output": "",
    "chat_history": [],
    "show_study_plan": False,
    "question_submitted": False,
    "last_video_url": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------- Functions ----------------------
def get_youtube_transcript(video_url):
    long_form = re.search(r"v=([a-zA-Z0-9_-]{11})", video_url)
    short_form = re.search(r"youtu.be/([a-zA-Z0-9_-]{11})", video_url)

    if long_form:
        video_id = long_form.group(1)
    elif short_form:
        video_id = short_form.group(1)
    else:
        raise ValueError("Invalid YouTube URL format.")

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(["en"])
        fetched = transcript.fetch()
    except TranscriptsDisabled:
        raise ValueError("Transcripts are disabled for this video.")
    except (NoTranscriptFound, CouldNotRetrieveTranscript):
        try:
            fetched = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e2:
            raise ValueError(f"Could not fetch transcript: {e2}")

    texts = [item.get("text", "").strip() for item in fetched]
    return " ".join(texts)


def remove_emojis(text):
    return re.sub(r"[^\x00-\x7F]+", "", text)


def generate_pdf(text, filename="summary_notes.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Calculate the available width for the cell, which is the page width minus double the left margin.
    # This ensures there's always a positive width for the text to be rendered.
    cell_width = pdf.w - 2 * pdf.l_margin
    if cell_width <= 0:
        raise FPDFException("Calculated PDF cell width is zero or negative. Check page dimensions or margins.")
    for line in text.split("\n"):
        pdf.multi_cell(cell_width, 10, line) # Use the calculated explicit width
    pdf.output(filename)
    return filename


def build_chat_prompt(summary, chat_history, new_question):
    prompt = f"""You are a helpful assistant answering questions based on the following lecture summary:

{summary}

"""
    if chat_history:
        prompt += "Here is the conversation so far:\n"
        for q, a in chat_history:
            prompt += f"Q: {q}\nA: {a}\n"
    prompt += f"\nQ: {new_question}\n\nAnswer:"
    return prompt


# ---------------------- Model Setup ----------------------
# Replace with summarization model for better results if available
text_generator = pipeline("text-generation", model="distilgpt2")


def generate_summary(transcript_text):
    prompt = f"""
You are a helpful study assistant.

Given the transcript of a lecture or tutorial:

{transcript_text}

Please:
1. Summarize it in bullet points.
2. Suggest smart, self-explanatory note headings and bullet points.

Format exactly as:

📋 Summary:
- Bullet 1
- Bullet 2

📝 Smart Notes:
Heading 1:
- Point A
- Point B

Heading 2:
- Point C
"""
    results = text_generator(prompt, max_length=500, num_return_sequences=1)
    return results[0]["generated_text"]


def generate_answer(summary, chat_history, new_question):
    prompt = build_chat_prompt(summary, chat_history, new_question)
    results = text_generator(prompt, max_length=200, num_return_sequences=1)
    answer = results[0]["generated_text"][len(prompt) :].strip()
    return answer


# ---------------------- Streamlit UI ----------------------
video_url = st.text_input(
    "📺 Enter YouTube lecture/tutorial URL:", value=st.session_state.last_video_url
)

if video_url and video_url != st.session_state.last_video_url:
    st.session_state.last_video_url = video_url
    st.session_state.summary_output = ""
    st.session_state.chat_history = []
    st.session_state.show_study_plan = False
    st.session_state.question_submitted = False

    with st.spinner("🔍 Fetching transcript..."):
        try:
            st.session_state.transcript_text = get_youtube_transcript(video_url)
            st.success("✅ Transcript fetched successfully!")
        except Exception as e:
            st.error(f"❌ Error fetching transcript: {e}")
            st.stop()

if st.session_state.transcript_text:
    st.text_area(
        "🧾 Transcript Preview:",
        st.session_state.transcript_text[:2000] + "...",
        height=200,
    )

if st.button("🧠 Summarize Transcript"):
    with st.spinner("⏳ Summarizing..."):
        try:
            output = generate_summary(st.session_state.transcript_text)
            clean_output = remove_emojis(output)
            st.session_state.summary_output = clean_output
            st.session_state.chat_history = []
            st.session_state.show_study_plan = False
            st.session_state.question_submitted = False
            st.success("✅ Summary and Notes Generated!")
        except Exception as e:
            st.error(f"❌ Summarization Error: {e}")

if st.session_state.summary_output:
    st.markdown("### 📋 Summary & 📝 Smart Notes")
    st.markdown(st.session_state.summary_output)

    pdf_file = generate_pdf(st.session_state.summary_output)
    with open(pdf_file, "rb") as file:
        st.download_button(
            label="💾 Download Summary as PDF",
            data=file,
            file_name="Smart_Lecture_Summary.pdf",
            mime="application/pdf",
        )

    st.divider()
    st.subheader("💬 Ask questions about the lecture summary")

    with st.form(key="chat_form", clear_on_submit=True):
        question = st.text_input(
            "Ask a question related to the lecture summary:", key="chat_input"
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

    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"Q: {q}")
        st.markdown(f"A: {a}")
        st.markdown("---")
