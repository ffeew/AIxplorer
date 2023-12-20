import streamlit as st
from langchain.llms import LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.utilities import GoogleSearchAPIWrapper
from dotenv import load_dotenv
import config
from langchain.callbacks.base import BaseCallbackHandler


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


@st.cache_resource
def st_load_retriever(_llm, mode):
    model_kwargs = {"device": "cuda"}
    embeddings_model = HuggingFaceEmbeddings(
        model_name=config.embeddings_model,
        model_kwargs=model_kwargs,
    )

    vector_store = Chroma(
        "cs_paper_store",
        embeddings_model,
        persist_directory=config.vector_db_path,
    )

    if mode == "vectordb":
        # load the vector store
        return vector_store.as_retriever()

    elif mode == "google search":
        load_dotenv()
        search = GoogleSearchAPIWrapper()
        web_research_retriever = WebResearchRetriever.from_llm(
            vectorstore=vector_store, llm=_llm, search=search
        )
        return web_research_retriever

    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")


@st.cache_resource
def st_load_llm(
    temperature=config.temperature,
    max_tokens=config.max_tokens,
    top_p=config.top_p,
    llm_path=config.llm_path,
    context_length=config.context_length,
    n_gpu_layers=config.n_gpu_layers,
    n_batch=config.n_batch,
):
    llm = LlamaCpp(
        model_path=llm_path,
        temperature=temperature,
        max_tokens=max_tokens,
        n_ctx=context_length,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        top_p=top_p,
        verbose=False,
    )

    return llm
