from langchain_classic.retrievers import MultiQueryRetriever


def get_retriever(
    vectorstore,
    llm=None,
    search_type: str = "similarity",
    k: int = 2,
    use_multi_query: bool = False
):
  

    base_retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )

    if use_multi_query:

        if llm is None:
            raise ValueError(
                "LLM must be provided when use_multi_query=True."
            )

        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )

    return base_retriever