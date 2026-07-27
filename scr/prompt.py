from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder


def get_rag_prompt():

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an intelligent AI assistant specialized in answering questions about a YouTube video.

Your task is to answer the user's question ONLY using the retrieved transcript context.

Guidelines:

1. Use only the provided context.
2. Do not make up information.
3. If the answer is not present in the context, say:
   "I couldn't find the answer in the provided video transcript."
4. Explain clearly and naturally.
5. If appropriate, answer using bullet points.
6. Keep answers concise but informative.
"""
            ),
            (
                "human",
                """
Context:
{context}

Question:
{question}

Answer:
"""
            )
        ]
    )

    return prompt


def get_history_aware_prompt():
  
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Given the chat history and the latest user question,
rewrite the question so that it can be understood without the chat history.

Do NOT answer the question.

Only rewrite it if necessary.
Otherwise return it unchanged.
"""
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    return prompt