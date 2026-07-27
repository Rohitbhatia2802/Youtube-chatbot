from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_transcript(
        document:list[Document],
        chunk_size:int=1000,
        chunk_overlap:int=200
) -> list[Document]:
    
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)

    split_docs=text_splitter.split_documents(document)

    return split_docs