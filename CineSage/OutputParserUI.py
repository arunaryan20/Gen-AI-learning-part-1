import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser


# ---------------- Pydantic Model ----------------
class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: Optional[str]
    rating: Optional[float]
    summary: Optional[str]


# ---------------- Page Config ----------------
st.set_page_config(page_title="Movie Extractor", page_icon="🎬", layout="centered")
st.title("🎬 Movie Information Extractor")
st.caption("Powered by Ollama · qwen2.5:7b · Pydantic Output Parser")


# ---------------- Cached Resources ----------------
@st.cache_resource
def load_model():
    return ChatOllama(model="qwen2.5:7b", temperature=0)


@st.cache_resource
def load_parser():
    return PydanticOutputParser(pydantic_object=Movie)


@st.cache_resource
def load_prompt():
    parser = load_parser()
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an information extraction and summarization assistant.

Analyze the paragraph provided by the user and extract the most useful information from it.

{format_instructions}

Rules:
- Extract information only from the provided paragraph.
- Do not invent information that is not present.
- Correct obvious spelling mistakes when the intended name or word is clear.
- If some information is not available, set the field to null (or use "Not mentioned" for text fields).
- Keep the summary to 2-3 sentences.
- Make the output valid JSON that strictly matches the schema.
- Do NOT include any explanation, comments, or text outside the JSON.
            """,
            ),
            ("human", "{text}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())


model = load_model()
parser = load_parser()
prompt = load_prompt()


# ---------------- Input ----------------
paragraph = st.text_area(
    "Enter your paragraph below:",
    height=200,
    placeholder="Paste the paragraph you want to extract movie information from...",
)


# ---------------- Extract Button ----------------
if st.button("Extract Information", use_container_width=True, type="primary"):
    if not paragraph.strip():
        st.warning("Please enter a paragraph first.")
    else:
        with st.spinner("Extracting information..."):
            messages = prompt.invoke({"text": paragraph})
            response = model.invoke(messages)

        # ---- Raw output (kept in an expander, like your print) ----
        with st.expander("🔍 View raw model output"):
            st.text(response.content)

        # ---- Parse + display ----
        try:
            movie: Movie = parser.parse(response.content)

            st.markdown("### 📌 Extracted Information")

            # Title
            st.subheader(f"🎬 {movie.title}")

            # Top metrics row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📅 Release Year", movie.release_year if movie.release_year else "N/A")
            with col2:
                st.metric("⭐ Rating", movie.rating if movie.rating is not None else "N/A")
            with col3:
                st.metric("🎭 Genres", len(movie.genre) if movie.genre else 0)

            st.markdown("---")

            # Director & Cast
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**🎬 Director**  \n{movie.director or 'Not mentioned'}")
            with col_b:
                st.markdown(f"**👥 Cast**  \n{movie.cast or 'Not mentioned'}")

            st.markdown("---")

            # Genres as pills
            if movie.genre:
                st.markdown("**🎭 Genres**")
                st.markdown(" ".join([f"`{g}`" for g in movie.genre]))

            # Summary
            if movie.summary:
                st.markdown("**📝 Summary**")
                st.info(movie.summary)

        except Exception as e:
            st.error("Failed to parse output into a Movie object.")
            st.exception(e)