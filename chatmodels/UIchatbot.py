import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    /* Mode badge */
    .mode-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    .mode-angry { background-color: #ffebee; color: #c62828; border: 1px solid #ef9a9a; }
    .mode-funny { background-color: #fff8e1; color: #f57f17; border: 1px solid #ffe082; }
    .mode-sad   { background-color: #e3f2fd; color: #1565c0; border: 1px solid #90caf9; }
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return ChatOllama(model="qwen2.5:7b")

model = load_model()

# ---------------- Mode Definitions ----------------
MODES = {
    "😡 Angry Mode": {
        "system": "You are an angry AI agent. You respond aggressively and impatiently",
        "css": "mode-angry"
    },
    "😂 Funny Mode": {
        "system": "You are a very funny AI agent. You respond with humor and jokes",
        "css": "mode-funny"
    },
    "😢 Sad Mode": {
        "system": "You are a sad AI agent. You respond with empathy and understanding",
        "css": "mode-sad"
    },
}

# ---------------- Sidebar: Mode Selection ----------------
with st.sidebar:
    st.markdown("### 🎭 Choose Your AI Mode")
    selected_mode = st.radio(
        "Select a personality:",
        options=list(MODES.keys()),
        index=1,
        label_visibility="collapsed"
    )
    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [SystemMessage(content=MODES[selected_mode]["system"])]
        st.session_state.current_mode = selected_mode
        st.rerun()

    st.markdown("---")
    st.caption("💡 Switch modes anytime — the chat will reset.")

# ---------------- Session State Initialization ----------------
if "current_mode" not in st.session_state:
    st.session_state.current_mode = selected_mode
    st.session_state.messages = [SystemMessage(content=MODES[selected_mode]["system"])]

# If user switches mode, reset the conversation with the new system prompt
if st.session_state.current_mode != selected_mode:
    st.session_state.current_mode = selected_mode
    st.session_state.messages = [SystemMessage(content=MODES[selected_mode]["system"])]
    st.rerun()

# ---------------- Header ----------------
st.markdown('<div class="main-title">🤖 AI Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Ollama · qwen2.5:7b</div>', unsafe_allow_html=True)

# Current mode badge
mode_css = MODES[selected_mode]["css"]
st.markdown(
    f'<div style="text-align:center;"><span class="mode-badge {mode_css}">'
    f'Current Mode: {selected_mode}</span></div>',
    unsafe_allow_html=True
)

# ---------------- Render Chat History ----------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🧑"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.write(msg.content)

# ---------------- Chat Input ----------------
prompt = st.chat_input("Type your message here...")

if prompt:
    # Show user message
    with st.chat_message("user", avatar="🧑"):
        st.write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Get and show assistant response (streamed)
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        full_response = ""
        for chunk in model.stream(st.session_state.messages):
            full_response += chunk.content
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    st.session_state.messages.append(AIMessage(content=full_response))