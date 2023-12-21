import streamlit as st
from utils import st_load_retriever, st_load_llm, StreamHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentType, initialize_agent, load_tools


RETRIEVAL_METHOD_MAP = {
    "Vector Database": "vectordb",
    "Google Search": "google search",
    "DuckDuckGo Search": "duckduckgo search",
}

st.title("AIxplorer - A Smarter Google Scholar 🌐📚")
st.write(
    "AIxplorer aims to revolutionize academic research by combining the capabilities of traditional search engines like Google Scholar with an advanced retrieval augmented generation (RAG) system. Built on Python and Langchain, this application provides highly relevant and context-aware academic papers, journals, and articles, elevating the standard of academic research."
)


st.divider()
st.subheader("Settings")
col1, col2, col3 = st.columns(3)

with col1:
    retrieval_method = st.selectbox(
        "Retrieval Mode",
        RETRIEVAL_METHOD_MAP.keys(),
        index=0,
        help="The retrieval method used to retrieve supporting documents.",
    )

st.divider()

llm = st_load_llm()

# first path
if retrieval_method in ("Vector Database", "Google Search"):
    retriever = st_load_retriever(llm, RETRIEVAL_METHOD_MAP[retrieval_method])
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

# second path
else:
    tools = load_tools(["ddg-search"])
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent.run(prompt, callbacks=[st_callback])
            st.write(response)
