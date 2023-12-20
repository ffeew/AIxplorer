vector_db_path = (
    "/home/daniel/Desktop/AIxplorer/chromadb"  # path to the vector database
)

embeddings_model = "BAAI/bge-small-en"  # embeddings model to use to generate vectors

llm_path = "/home/daniel/Desktop/AIxplorer/mistral-7b-openorca.Q5_K_M.gguf"  # path to the LLM model

device = "cuda"  # device to use for the LLM model, "cuda" or "cpu

n_gpu_layers = 50  # Change this value based on your model and your GPU VRAM pool. Change to 0 if you are using a CPU.

n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

context_length = 8000  # length of the context to use for the LLM model

temperature = 0.0  # temperature to use for the LLM model

top_p = 1.0  # top_p to use for the LLM model

max_tokens = 2000  # maximum number of tokens to generate from the LLM model
