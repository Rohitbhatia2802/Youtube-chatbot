from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda

from langchain_classic.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage


from loader import load_transcript
from splitter import split_transcript
from embedding import get_embedding_model
from vectorstore import get_vectorstore
from retriever import get_retriever
from prompt import (
    get_rag_prompt,
    get_history_aware_prompt,
)

load_dotenv()



llm = ChatOpenAI(
    model="deepseek/deepseek-chat-v3-0324",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
    max_tokens=256
)

parser=StrOutputParser()


def format_docs(docs):
    
    formatted_context=[]

    for index,doc in enumerate(docs,start=1):
        formatted_context.append(f""" 
                                 
                        chunk {index}
                    ------------------------------
                       {doc.page_content}
 """ )

    return "\n======================\n".join(formatted_context)

def build_history_aware_retriever(retriever, llm):
   
    history_prompt = get_history_aware_prompt()

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=history_prompt
    )

    return history_aware_retriever


def build_rag_chain(history_aware_retriever):
    
    rag_prompt= get_rag_prompt()

    rag_chain = (
        RunnablePassthrough.assign(
            context=(
                RunnableLambda(
                    lambda x: {
                        "input": x["question"],
                        "chat_history": x["chat_history"],
                    }
                )
                | history_aware_retriever
                | format_docs
            )
        )
        | rag_prompt
        | llm
        | parser
    )

    return rag_chain

def process_video(youtube_url: str):
    """
    Processes a YouTube video and returns a ready-to-use RAG chain.
    """

    
    document, video_id = load_transcript(youtube_url)

    
    chunks = split_transcript([document])

    
    embedding_model = get_embedding_model()

    
    vectorstore = get_vectorstore(
        documents=chunks,
        embedding_model=embedding_model,
        video_id=video_id
    )

    
    retriever = get_retriever(
        vectorstore=vectorstore
    )

    
    history_aware_retriever = build_history_aware_retriever(
        retriever=retriever,
        llm=llm
    )

    # Build LCEL RAG chain
    rag_chain = build_rag_chain(
        history_aware_retriever
    )

    return rag_chain


def answer_question(
    rag_chain,
    question: str,
    chat_history: list
):
    """
    Uses the existing RAG chain to answer a user's question.
    """

    answer = rag_chain.invoke(
        {
            "question": question,
            "chat_history": chat_history
        }
    )

    return answer


if __name__ == "__main__":

    url = input("Enter YouTube URL: ")

    # Process video only once
    rag_chain = process_video(url)

    chat_history = []

    while True:

        question = input("\nYou: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        answer = answer_question(
            rag_chain=rag_chain,
            question=question,
            chat_history=chat_history
        )

        print(f"\nBot: {answer}")

        # Update chat history
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))



