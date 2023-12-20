import options as opt
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from utils.retriever import load_retriever


def main():
    # instantiate the LLM
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = LlamaCpp(
        model_path=opt.llm_path,
        temperature=opt.temperature,
        max_tokens=opt.max_tokens,
        n_ctx=opt.context_length,
        n_gpu_layers=opt.n_gpu_layers,
        n_batch=opt.n_batch,
        top_p=opt.top_p,
        callback_manager=callback_manager,
        verbose=False,
    )

    retriever = load_retriever(llm)

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=retriever)

    while True:
        user_input = input("Enter question: ")
        result = qa_chain({"question": user_input})
        print(result, end="\n\n")


if __name__ == "__main__":
    main()
