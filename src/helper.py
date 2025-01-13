from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

def load_pdf_file(data):
    '''
    Parameters: 
        data: a string containing the folder location for the data

    Returns:
        Data scanned from pdf
    '''
    loader = DirectoryLoader(data, 
                             glob = "*.pdf",
                             loader_cls = PyPDFLoader)
    
    documents = loader.load()
    return documents

def text_split(extracted_data):
    '''
    Parameters:
        extracted_data: data extracted from the pdf

    Returns:
        Data converted to text chunks form
    '''
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_hugging_face_embeddings():
    '''
    Function to download the embedding model from huggingface

    '''
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embedding


