from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Use a smaller/faster Hugging Face embedding model for Streamlit Cloud
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_vector_store(chunks):
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return vector_store
