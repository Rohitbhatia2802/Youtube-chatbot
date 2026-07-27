import os
from langchain_chroma import Chroma


DB_DIRECTORY = "db"


def get_db_path(video_id: str) -> str:
    
    return os.path.join(DB_DIRECTORY, video_id)


def create_vectorstore(documents, embedding_model, video_id: str):

    db_path = get_db_path(video_id)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=db_path
    )

    return vectorstore


def load_vectorstore(embedding_model, video_id: str):

    db_path = get_db_path(video_id)

    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model
    )

    return vectorstore


def get_vectorstore(documents, embedding_model, video_id: str):
    
    db_path = get_db_path(video_id)

    if os.path.exists(db_path) and os.listdir(db_path):
        print(f"Loading existing database for {video_id}")

        return load_vectorstore(
            embedding_model,
            video_id
        )

    print(f"Creating database for {video_id}")

    return create_vectorstore(
        documents,
        embedding_model,
        video_id
    )