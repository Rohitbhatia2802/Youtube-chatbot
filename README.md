# Conversational YouTube Assistant

A local, Retrieval-Augmented Generation (RAG) chatbot for YouTube videos. Paste a YouTube URL, let the app fetch and index its English transcript, then ask questions grounded in that transcript.

## Features

- Retrieves English captions/transcripts from YouTube videos
- Splits transcripts into searchable chunks and embeds them with `all-MiniLM-L6-v2`
- Persists a separate Chroma vector database per video
- Uses conversational history to interpret follow-up questions
- Provides a Streamlit chat interface with saved in-session conversations
- Refuses to invent answers when information is absent from the transcript

## Prerequisites

- Python 3.10 or newer
- [Ollama](https://ollama.com/) installed and running locally
- The `llama3.2` model downloaded:

```powershell
ollama pull llama3.2
```

The video must have an English transcript available. Videos without captions, private videos, or transcripts blocked by YouTube cannot be processed.

## Installation

From the project root, create and activate a virtual environment (optional but recommended):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the dependencies, including the two LangChain packages used by the application:

```powershell
pip install -r requirements.txt
pip install langchain-ollama langchain-classic
```

## Run the web app

The application source lives in `scr`, so start Streamlit from that folder:

```powershell
cd scr
streamlit run app.py
```

Open the local URL shown by Streamlit, paste a YouTube URL in the sidebar, select **Process Video**, and begin chatting.

## Run from the command line

An interactive terminal version is also available:

```powershell
cd scr
python rag_chain.py
```

Enter a YouTube URL when prompted, ask questions, and type `exit` to quit.

## How it works

1. The app extracts the YouTube video ID and fetches its English transcript.
2. The transcript is split into overlapping 1,000-character chunks.
3. Each chunk is embedded with the Hugging Face `sentence-transformers/all-MiniLM-L6-v2` model.
4. Chroma saves the vectors under `scr/db/<video-id>/` and reuses them on later runs.
5. For each question, the app retrieves the four most similar chunks and asks local `llama3.2` to answer using only that context.

## Project structure

```text
scr/
|-- app.py           # Streamlit interface
|-- rag_chain.py     # RAG orchestration and CLI entry point
|-- loader.py        # YouTube URL parsing and transcript loading
|-- splitter.py      # Transcript chunking
|-- embedding.py     # Hugging Face embedding model
|-- vectorstore.py   # Persistent Chroma storage
|-- retriever.py     # Similarity retriever
|-- prompt.py        # Answering and query-rewrite prompts
`-- db/              # Generated local vector databases
```

## Notes

- Vector databases in `scr/db/` are generated artifacts and may grow as more videos are processed.
- The current application does not require an OpenAI API key; generation runs through local Ollama.
- Chat history is stored only in Streamlit's current browser session.
