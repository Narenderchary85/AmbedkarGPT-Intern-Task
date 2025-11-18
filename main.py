from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


def build_vector_db():
    print("Loading text from speech.txt")
    loader = TextLoader("speech.txt")
    documents = loader.load()

    print("Splitting text into chunks")
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=400,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    print("Generating embeddings")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Storing vectors in Chroma")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory="db"
    )
    vectordb.persist()
    print("Vector DB created successfully")

    return vectordb


def load_vector_db():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(
        persist_directory="db",
        embedding_function=embedding
    )


def create_qa_chain():
    vectordb = load_vector_db()

    print("Loading Mistral 7B using Ollama")
    llm = Ollama(model="mistral")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )

    return qa


def main():
    print("=== AmbedkarGPT — Q&A System ===\n")
    try:
        load_vector_db()
        print("Existing DB found. Loading")
    except:
        print("No DB found. Creating new one")
        build_vector_db()

    qa = create_qa_chain()

    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() == "exit":
            break

        result = qa.invoke({"query": question})
        print("\n Answer:\n", result["result"])



if __name__ == "__main__":
    main()
