from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Database location
db_location = "./chrome_langchain_db"

# Embeddings (MUST be provided for querying)
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

# Load the vector store with the embedding function
vector_store = Chroma(
    collection_name="document_chunks",
    persist_directory=db_location,
    embedding_function=embeddings  # This is required!
)

# Set up retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    query = input("\n🔍 Enter your search query: ")
    results = retriever.invoke(query)

    if results:
        print("\n📄 **Search Results:**\n")
        for result in results:
            print(result.page_content)
            print(f"📌 Source: {result.metadata['source']}\n")
    else:
        print("⚠️ No matching results found.")
