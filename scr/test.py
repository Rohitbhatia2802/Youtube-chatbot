from loader import load_transcript
from splitter import split_transcript
from embedding import get_embedding_model
from vectorstore import get_vectorstore
from retriever import get_retriever

url = "https://www.youtube.com/watch?v=4l97aNza_Zc"

doc = load_transcript(url)

chunks = split_transcript([doc])

embedding_model = get_embedding_model()

# Create or load vector database
vectorstore = get_vectorstore(
    documents=chunks,
    embedding_model=embedding_model,
    video_id=doc.metadata["video_id"]
)

retriever=get_retriever(vectorstore)

query="What is feastable?"

docs=retriever.invoke(query)
print(docs)