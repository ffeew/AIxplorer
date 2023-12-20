import options as opt
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.utilities import GoogleSearchAPIWrapper
from dotenv import load_dotenv


def load_retriever(llm):
    model_kwargs = {"device": opt.device}
    embeddings_model = HuggingFaceEmbeddings(
        model_name=opt.embeddings_model,
        model_kwargs=model_kwargs,
    )

    vector_store = Chroma(
        "cs_paper_store",
        embeddings_model,
        persist_directory=opt.vector_db_path,
    )

    if opt.retrieval_mode == "vectordb":
        # load the vector store
        return vector_store.as_retriever()

    elif opt.retrieval_mode == "google search":
        load_dotenv()
        search = GoogleSearchAPIWrapper()
        web_research_retriever = WebResearchRetriever.from_llm(
            vectorstore=vector_store, llm=llm, search=search
        )
        return web_research_retriever

    else:
        raise ValueError(f"Unknown retrieval mode: {opt.retrieval_mode}")
