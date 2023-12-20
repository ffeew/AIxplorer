import streamlit as st
from utils import st_load_retriever, st_load_llm, StreamHandler
from langchain.chains import RetrievalQAWithSourcesChain

st.title("AIxplorer - A Smarter Google Scholar 🌐📚")
st.write(
    "AIxplorer aims to revolutionize academic research by combining the capabilities of traditional search engines like Google Scholar with an advanced retrieval augmented generation (RAG) system. Built on Python and Langchain, this application provides highly relevant and context-aware academic papers, journals, and articles, elevating the standard of academic research."
)


st.divider()
st.subheader("Settings")
col1, col2, col3 = st.columns(3)

with col1:
    use_google = st.checkbox(
        "Use Google Search",
        value=True,
        help="Use Google Search to retrieve papers. If unchecked, will use the vector database.",
    )
st.divider()

llm = st_load_llm()
retriever = st_load_retriever(llm, "vectordb" if not use_google else "google search")

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=retriever)

user_input = st.text_area(
    "Enter your query here",
    help="Query should be on computer science as the RAG system is tuned to that domain.",
)


if st.button("Generate"):
    st.divider()
    st.subheader("Answer:")
    with st.spinner("Generating..."):
        container = st.empty()
        stream_handler = StreamHandler(container)
        response = qa_chain({"question": user_input}, callbacks=[stream_handler])
