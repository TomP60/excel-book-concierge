import streamlit as st 
from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
import faiss
import json
import numpy as np

# --- Load book search index ---
index = faiss.read_index("book_pdf_index.faiss")
with open("book_pdf_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# For local testing
#load_dotenv("GPT35.env")
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# For Streamlit Cloud
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding, dtype='float32')

def search_book_chunks(query, k=3):
    query_vec = get_embedding(query).reshape(1, -1)
    D, I = index.search(query_vec, k)
    return [metadata[i] for i in I[0]]

# --- UI tweaks ---
def get_base64_image(path):
    with open(path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
        return f"data:image/jpeg;base64,{encoded}"

st.markdown("""
    <style>
        input, textarea {
            border: 1px solid #ccc !important;
            background-color: #fafafa !important;
            padding: 8px !important;
            border-radius: 4px !important;
        }
        .output-box {
            border: 1px solid #ddd;
            background-color: #fdfdfd;
            padding: 10px;
            border-radius: 4px;
            min-height: 100px;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# --- Page config ---
st.set_page_config(
    page_title="Book Concierge – Mastering Excel",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Branding ---
logo_base64 = get_base64_image("PetiteKatPress logo Thumb 200.jpg")
st.markdown(f"""
    <div style='display: flex; align-items: center; gap: 12px;'>
        <img src="{logo_base64}" width="100"/>
        <h3 style='margin: 0;'>Book Concierge: Mastering Excel</h3>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

cover_base64 = get_base64_image("MasteringExcel.PNG")
st.markdown(f"""
    <div style='display: flex; align-items: center; gap: 20px;'>
        <a href="https://www.amazon.com/dp/B0FF377S6T" target="_blank">
            <img src="{cover_base64}" width="160" alt="Mastering Excel Book Cover"/>
        </a>
        <div>
            <h4>Welcome to the <strong>Book Concierge</strong></h4>
            <p>I'm here to help you decide whether <em>Mastering Excel for Home Budgeting</em> is right for you.</p>
            <p>Ask anything about the book — what's covered, who it's for, why it also uses LibreCalc, or how it might help your budgeting journey.</p>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")


# --- Session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

# --- Input ---

user_input = st.text_input("🔍 Ask a question about the book (in any language):")
output_placeholder = st.empty()

if user_input:
    with st.spinner("Looking through the book…"):
        matches = search_book_chunks(user_input, k=3)
        context_text = "\n\n".join([f"[Page {m['page']}]:\n{m['text']}" for m in matches])

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Book Concierge for 'Mastering Excel for Home Budgeting'.\n"
                    "You ONLY answer questions based on the content of the book.\n"
                    "If a user asks about unrelated topics, politely explain that your purpose is to help them understand this book.\n"
                    "If the user expresses interest in buying, you may ask: 'Would you prefer PDF, Kindle, or Paperback?'\n"
                    "Provide answers in plain language, try to answer questions in the same language the user used. Default to English if you cannot.\n"
                    "If the user seems unsure about the book, you may suggest helpful prompts like:\n"
                    "- Would you like to know what topics are covered?\n"
                    "- Do you want to see examples of how Excel is used in budgeting?\n"
                    "- Would it help if I explained what kind of reader this book is best suited for?\n"
                    "Responses to all questions about support you should also mention the Companion Agent is there to help and is free to owners of the book.\n"
                    "Format all answers in clear, readable Markdown with short paragraphs, bullet points if helpful, and line breaks between ideas.\n"
                    "Do not push a sale or redirect — your goal is to inform, not convert."
                )
            }
        ]

        # --- Add prior chat history to context ---
        for prior in st.session_state.chat_history:
            messages.append({"role": "user", "content": prior["user"]})
            messages.append({"role": "assistant", "content": prior["assistant"]})

        # --- Add current user question with book excerpts ---
        messages.append({
            "role": "user",
            "content": f"Book excerpts:\n\n{context_text}\n\nQuestion: {user_input}"
        })

        if st.session_state.question_count >= 10:
            st.warning("⚠️ You've reached the 10-question limit for this session. Please refresh the page to start a new conversation.")
            st.stop()

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        answer = response.choices[0].message.content
        output_placeholder.markdown(f"<div class='output-box'>{answer}</div>", unsafe_allow_html=True)

        # --- Save this exchange to memory ---
        st.session_state.chat_history.append({
            "user": user_input,
            "assistant": answer
        })
        st.session_state.question_count += 1

        with st.expander("🔎 View book excerpts used to answer your question"):
            for m in matches:
                st.markdown(f"**Page {m['page']}**")
                st.markdown(m['text'])
else:
    output_placeholder.markdown("<div class='output-box'>Waiting for your question…</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown(
    "Made by Thomas W. Pettit •  [PetiteKat Press](https://petitekatpress.com)  ",
    unsafe_allow_html=True
)
