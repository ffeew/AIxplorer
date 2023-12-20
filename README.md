# AIxplorer: A Retrieval Augmented Generation Application for Research

## Introduction

AIxplorer aims to revolutionize academic research by combining the capabilities of traditional search engines like Google Scholar with an advanced retrieval augmented generation (RAG) system. Built on Python and Langchain, this application provides highly relevant and context-aware academic papers, journals, and articles, elevating the standard of academic research.

## Features

- **Context-Aware Search**: Get highly relevant search results tailored to your specific query.
- **Summarization**: Automatic summarization of research papers for quick insights.
- **Citation Helper**: Quickly generate citations in multiple formats.
- **State-of-the-Art Language Models**: The system is powered by Langchain and llama-cpp, integrating large language models to interpret, augment, and respond to user queries with unprecedented precision.

## Installation

### Prerequisites

Make sure you have the following installed on your system:

1. Python 3.10 or higher
2. (Optional) Cuda toolkit if you intend to use a GPU to speed up the inference process.

### Steps

1. Clone the repository:

```bash
git clone https://github.com/ffeew/AIxplorer.git
```

2. Navigate to the directory:

```bash
cd AIxplorer
```

3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate the virtual environment:

```bash
source venv/bin/activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Obtain the CS paper dataset from [here](https://www.kaggle.com/datasets/ffeewww/arvix-sample-dataset). Note that this is a just a sample dataset intended as a proof of concept.

7. Extract the dataset contents in the `dataset` folder.

8. Set up the vector database by running the vector database builder script:

```bash
python vector_db_builder.py
```

9. Download a LLM model in the gguf format and update the `options.py` file with the path to the model. You can download the model from huggingface. My recommendation is [Mistral-7B-OpenOrca](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF).

#### Optional

To enable web search retrieval:

1. Go to https://developers.google.com/custom-search/v1/introduction and click on "Get a Key".
2. Create a new project and enable the Custom Search API.
3. Create a programmable search engine from [here](https://programmablesearchengine.google.com/controlpanel/create) and copy the search engine ID.
4. Using the .env.example file, create a .env file and fill in the API key and search engine ID.
5. For more information on how to get the API key and search engine ID, refer to [this](https://python.langchain.com/docs/modules/data_connection/retrievers/web_research) page.

## Usage

1. Activate the virtual environment:

```bash
source venv/bin/activate
```

2. Run the main script:

```bash
python main.py
```

## Technology Stack

- **Python**: Core programming language
- **Langchain**: Used for the Retrieval Augmented Generation model
- Additional Libraries: `llama-cpp`, `chromadb` and more.
- Check out the `requirements.txt` file for a full list of dependencies.

## Contributions

Pull requests and issues are welcome!

## License

This project is licensed under the MIT License.
