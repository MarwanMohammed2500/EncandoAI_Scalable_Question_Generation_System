from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def read_pdf(file_path):
    """
    This function loads PDF Files
    Arguments:
    file_path: str -> The path for the file
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    contents = [doc.page_content for doc in documents]
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.create_documents(contents)
    return chunks

def read_dot_txt(file_path):
    """
    This function loads TXT Files
    Arguments:
    file_path: str -> The path for the file
    """
    loader = TextLoader(file_path)
    documents = loader.load()
    contents = [doc.page_content for doc in documents]
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.create_documents(contents)
    return chunks