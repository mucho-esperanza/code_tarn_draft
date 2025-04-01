from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os

# Directory containing documents
documents_dir = "./Documents"  # Change this to your directory

# Database location
db_location = "./chrome_langchain_db"

# Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

# Initialize or load the vector store
vector_store = Chroma(
    collection_name="document_chunks",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Track already processed files
existing_docs = set(vector_store.get()["ids"])  # Fetch existing document IDs
processed_files = set()

# Function to process a single document
def process_document(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    # Generate a unique prefix for the document
    id_prefix = os.path.basename(file_path).replace(" ", "_")

    # Check if the document has already been processed
    if any(id_prefix in doc_id for doc_id in existing_docs):
        processed_files.add(file_path)
        return

    # Load the file based on its extension
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        print(f"⚠️ Unsupported file type: {file_path}")
        return

    documents = loader.load()

    # Split the documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Assign unique IDs
    ids = [f"{id_prefix}_{i}" for i in range(len(chunks))]

    # Add metadata
    for chunk in chunks:
        chunk.metadata["source"] = file_path

    # Add to vector store
    vector_store.add_documents(documents=chunks, ids=ids)
    print(f"✅ Added {len(chunks)} chunks from: {file_path}")

# Process all documents in the directory
new_files_added = False

for filename in os.listdir(documents_dir):
    file_path = os.path.join(documents_dir, filename)
    process_document(file_path)

    if file_path not in processed_files:
        new_files_added = True

# Print summary
print("\n📂 **Summary**")
if processed_files:
    print("📌 Already processed files:")
    for f in processed_files:
        print(f"  - {f}")
else:
    print("✅ No previously processed files found.")

if new_files_added:
    print("✅ New documents added to the vector store.")
else:
    print("🔹 No new documents were added.")

