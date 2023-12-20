sample_size = 5000  # number of papers to sample from the dataset.

dataset_path = "./dataset"  # path to the dataset

chunk_size = 200  # number of tokens per chunk to split the paper into

batch_size = 6000  # number of documents to process in a batch

vector_db_path = "./chromadb"  # path to the vector database

embeddings_model = "BAAI/bge-small-en"  # embeddings model to use to generate vectors

llm_path = "./mistral-7b-openorca.Q5_K_M.gguf"  # path to the LLM model

device = "cuda"  # device to use for the LLM model, "cuda" or "cpu

retrieval_mode = (
    "vectordb"  # location to retrieve data from, "vectordb" or "google search"
)

n_gpu_layers = 50  # Change this value based on your model and your GPU VRAM pool. Change to 0 if you are using a CPU.

n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

context_length = 8000  # length of the context to use for the LLM model

temperature = 0.0  # temperature to use for the LLM model

top_p = 1  # top_p to use for the LLM model

max_tokens = 2000  # maximum number of tokens to generate from the LLM model
