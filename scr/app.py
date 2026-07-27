import streamlit as st
import time

from langchain_core.messages import HumanMessage, AIMessage

from rag_chain import process_video, answer_question


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Conversational YouTube Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.main{
    padding-top:1rem;
}

.block-container{
    padding-top:2rem;
}

.stChatMessage{
    border-radius:12px;
    padding:10px;
}

[data-testid="stSidebar"]{
    background-color:#111827;
}

div[data-testid="stStatusWidget"]{
    border-radius:12px;
}

</style>
""",unsafe_allow_html=True)

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain=None

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

if "video_processed" not in st.session_state:
    st.session_state.video_processed=False

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

if "current_video" not in st.session_state:
    st.session_state.current_video=""

if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "youtube_input" not in st.session_state:
    st.session_state.youtube_input = ""

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🎥 Conversational YouTube Assistant")

st.caption(
    "Chat with any YouTube video using Retrieval-Augmented Generation (RAG)"
)

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("📂 Chats")

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        # Save current chat
        if (
            st.session_state.video_processed
            and len(st.session_state.chat_history) > 0
        ):

            chat_id = f"chat_{len(st.session_state.conversations)+1}"

            st.session_state.conversations[chat_id] = {

                "title": f"Chat {len(st.session_state.conversations)+1}",

                "video_url": st.session_state.current_video,

                "rag_chain": st.session_state.rag_chain,

                "chat_history": st.session_state.chat_history.copy()

            }

        # Start new chat
        st.session_state.chat_history = []
        st.session_state.rag_chain = None
        st.session_state.video_processed = False
        st.session_state.current_video = ""
        st.session_state.youtube_input = ""

        st.rerun()

    # -----------------------------
    # Everything below is OUTSIDE the button
    # -----------------------------

    st.divider()

    st.subheader("🎥 Process New Video")

    youtube_url = st.text_input(
        "Paste YouTube URL",
        key="youtube_input"
    )

    process_video_button = st.button(
        "🚀 Process Video",
        use_container_width=True
    )

    st.divider()

    for chat_id, chat in st.session_state.conversations.items():

        if st.button(chat["title"]):

            st.session_state.chat_history = chat["chat_history"]
            st.session_state.rag_chain = chat["rag_chain"]
            st.session_state.current_video = chat["video_url"]
            st.session_state.video_processed = True

            st.rerun()

# --------------------------------------------------
# Video Processing
# --------------------------------------------------

if process_video_button:

    if youtube_url.strip() == "":

        st.sidebar.error("⚠ Please enter a YouTube URL.")

    else:

        status = st.status(
            "🎥 Processing YouTube Video...",
            expanded=True
        )

        try:

            status.write("📥 Downloading transcript...")
            time.sleep(0.5)

            status.write("✂ Splitting transcript into chunks...")
            time.sleep(0.5)

            status.write("🧠 Creating embeddings...")
            time.sleep(0.5)

            status.write("📚 Building vector database...")
            time.sleep(0.5)

            status.write("🤖 Initializing chatbot...")
            time.sleep(0.5)


            st.session_state.rag_chain = process_video(youtube_url)
            st.session_state.video_processed = True
            st.session_state.current_video = youtube_url
            st.session_state.chat_history = []

            status.update(
                label="✅ Video processed successfully!",
                state="complete"
            )

            st.success("🎉 Your AI assistant is ready!")

        except Exception as e:

            status.update(
                label="❌ Failed to process video",
                state="error"
            )

            st.error(e)

# --------------------------------------------------
# Main Chat Window
# --------------------------------------------------

if st.session_state.video_processed:

    st.subheader("💬 Conversation")

    st.info(
        "Ask any question related to the processed YouTube video."
    )

    # Display existing conversation

    for message in st.session_state.chat_history:

        if isinstance(message, HumanMessage):

            with st.chat_message("user"):

                st.markdown(message.content)

        else:

            with st.chat_message("assistant"):

                st.markdown(message.content)

    # Chat Input

    question = st.chat_input(
        "Ask your question..."
    )

    if question:

        # Display User Message

        with st.chat_message("user"):

            st.markdown(question)

        # Assistant Response

        with st.chat_message("assistant"):

            answer_placeholder = st.empty()

            with st.spinner("Thinking..."):

                answer = answer_question(
                    rag_chain=st.session_state.rag_chain,
                    question=question,
                    chat_history=st.session_state.chat_history
                )

            # Fake streaming effect

            streamed_text = ""

            for word in answer.split():

                streamed_text += word + " "

                answer_placeholder.markdown(streamed_text + "▌")

                time.sleep(0.02)

            answer_placeholder.markdown(streamed_text)

        # Save Conversation

        st.session_state.chat_history.append(
            HumanMessage(content=question)
        )

        st.session_state.chat_history.append(
            AIMessage(content=answer)

        )

else:

    st.info(
        "👈 Process a YouTube video from the sidebar to start chatting."
    )

