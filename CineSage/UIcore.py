import re
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ---------------- Page Config ----------------
st.set_page_config(page_title="Info Extractor", page_icon="📄", layout="centered")
st.title("📄 Information Extraction Assistant")
st.caption("Powered by Ollama · qwen2.5:7b")

# ---------------- Load Model & Prompt ----------------
@st.cache_resource
def load_model():
    return ChatOllama(model="qwen2.5:7b", temperature=0)

@st.cache_resource
def load_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an information extraction and summarization assistant.

Analyze the paragraph provided by the user and extract the most useful information from it.

IMPORTANT FORMATTING RULES:
- Output ONE field per line.
- Put a SINGLE BLANK LINE between each field block.
- Do NOT merge multiple fields on the same line.
- Use the exact field names listed below, followed by a colon.

Format (follow exactly):

Title: <value>

Director: <value>

Release Year: <value>

Genre: <value>

Cast: <value>

Plot: <value>

Setting: <value>

Themes: <value>

Soundtrack: <value>

IMDb Rating: <value>

Reception: <value>

Key Points:
- <point 1>
- <point 2>
- <point 3>

Quick Summary: <2-3 sentence summary>

Extraction rules:
- Extract information only from the provided paragraph.
- Do not invent information that is not present.
- Correct obvious spelling mistakes when the intended name or word is clear.
- If some information is not available, write "Not mentioned".
- Keep the Plot concise.
- Keep the Quick Summary to 2-3 sentences.
            """
        ),
        ("human", "{text}")
    ])

model = load_model()
prompt = load_prompt()

# ---------------- Input ----------------
paragraph = st.text_area(
    "Enter your paragraph below:",
    height=200,
    placeholder="Paste the paragraph you want to extract information from..."
)

# ---------------- Field parser ----------------
FIELDS = [
    "Title", "Director", "Release Year", "Genre", "Cast",
    "Plot", "Setting", "Themes", "Soundtrack",
    "IMDb Rating", "Reception", "Key Points", "Quick Summary",
]

def parse_output(text: str) -> dict:
    """Split model output into {field_name: value} pairs."""
    # Build regex: match "Field:" up to the next "Field:" or end
    pattern = r"(?i)(" + "|".join(re.escape(f) for f in FIELDS) + r")\s*:\s*"
    matches = list(re.finditer(pattern, text))
    result = {}

    if not matches:
        return {"_raw": text.strip()}

    for i, m in enumerate(matches):
        field = m.group(1).strip().title()  # normalize case
        # Fix known case mismatches (e.g. "Imdb Rating" -> "IMDb Rating")
        for f in FIELDS:
            if f.lower() == field.lower():
                field = f
                break
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = text[start:end].strip()
        result[field] = value

    return result


def render_output(parsed: dict):
    """Render parsed dict as a clean, styled layout."""
    if "_raw" in parsed:
        st.markdown(parsed["_raw"])
        return

    # Header: Title + key facts as metrics
    if "Title" in parsed:
        st.subheader(f"🎬 {parsed['Title']}")

    cols = st.columns(3)
    with cols[0]:
        st.metric("📅 Release Year", parsed.get("Release Year", "—"))
    with cols[1]:
        st.metric("🎭 Genre", parsed.get("Genre", "—"))
    with cols[2]:
        st.metric("⭐ IMDb Rating", parsed.get("IMDb Rating", "—"))

    st.markdown("---")

    # Two-column layout for short fields
    left, right = st.columns(2)
    with left:
        st.markdown(f"**🎬 Director**  \n{parsed.get('Director', '—')}")
        st.markdown(f"**🎵 Soundtrack**  \n{parsed.get('Soundtrack', '—')}")
        st.markdown(f"**🌍 Setting**  \n{parsed.get('Setting', '—')}")
    with right:
        st.markdown(f"**👥 Cast**  \n{parsed.get('Cast', '—')}")
        st.markdown(f"**💡 Themes**  \n{parsed.get('Themes', '—')}")

    st.markdown("---")

    if "Plot" in parsed:
        st.markdown("**📖 Plot**")
        st.info(parsed["Plot"])

    if "Reception" in parsed:
        st.markdown("**📰 Reception**")
        st.write(parsed["Reception"])

    if "Key Points" in parsed:
        st.markdown("**🔑 Key Points**")
        kp = parsed["Key Points"]
        # Normalize bullet separators: split on newlines or "- "
        bullets = [b.strip(" -•\t") for b in re.split(r"\n|(?:^|\s)-\s", kp) if b.strip(" -•\t")]
        for b in bullets:
            st.markdown(f"- {b}")

    if "Quick Summary" in parsed:
        st.markdown("**📝 Quick Summary**")
        st.success(parsed["Quick Summary"])


# ---------------- Extract Button ----------------
if st.button("Extract Information", use_container_width=True, type="primary"):
    if not paragraph.strip():
        st.warning("Please enter a paragraph first.")
    else:
        with st.spinner("Extracting information..."):
            messages = prompt.invoke({"text": paragraph})
            response = model.invoke(messages)

        st.markdown("### 📌 Extracted Information")
        parsed = parse_output(response.content)
        render_output(parsed)

        # Optional: keep raw text in an expander for debugging
        with st.expander("🔍 View raw model output"):
            st.text(response.content)