from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
import os
from dotenv import load_dotenv
load_dotenv()
google_api = os.getenv("GOOGLE_API_KEY")

def create_embeddings_model(out_dim, model):
    """
    This function simply calls the Google Gemini Embeddings model with custom output dimensionality
    Arguments:
    out_dim: int -> The output dimensionality (dimensionality of the output vector)
    model: str -> The name of the model to use
    """
    return GoogleGenerativeAIEmbeddings(model=model, output_dimensionality=out_dim)

def store_vector(documents, embeddings_model):
    """
    This function takes on some document and stores it in a vector database
    Arguments:
    documents: Document -> The document to use
    embeddings_model: The embeddings model to use
    """
    vector_store = InMemoryVectorStore.from_documents(documents, embeddings_model)
    return vector_store