from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import pickle

## Load Ollama LAMA2 LLM model
llm = Ollama(model="llama2")

loader = PyPDFLoader("book.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = text_splitter.split_documents(docs)

db = FAISS.from_documents(docs, OllamaEmbeddings())

# Save the FAISS index
db.save_local("faiss_index1")



print("FAISS index and embeddings model saved.")
