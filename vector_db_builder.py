import glob
import random
import options as opt
from tqdm import tqdm
import ast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


all_files = glob.glob(f"{opt.dataset_path}/*.txt")
sampled_files = random.sample(all_files, opt.sample_size)
print(
    f"Sampling {len(sampled_files)} files from {len(all_files)} files in the dataset."
)

papers = []
metadata = []
failed = []

for file in tqdm(sampled_files):
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = f.read().split("\n\n\n\n\n")
            meta = ast.literal_eval(data[0])
            file_id = file.split("/")[-1]
            meta["source"] = f"http://arxiv.org/abs/{file_id.replace('.txt', '')}"
            content = data[1]
            papers.append(content)
            metadata.append(meta)
    except:
        print(f"Failed to process {file}")
        failed.append(file)


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=opt.chunk_size,
    chunk_overlap=0,
    disallowed_special=(),
)

print(f"Splitting {len(papers)} papers into chunks of {opt.chunk_size} characters.")
texts = text_splitter.create_documents(papers, metadatas=metadata)

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

print(f"Adding {len(texts)} documents to the vector store.")


def batch_add_documents(doc_list: list, batch_size: int):
    total_docs = len(doc_list)
    with tqdm(total=total_docs, desc="Adding documents", unit="docs") as progress_bar:
        for i in range(0, total_docs, batch_size):
            batch = doc_list[i : i + batch_size]
            vector_store.add_documents(batch)
            progress_bar.update(len(batch))


batch_add_documents(texts, opt.batch_size)
