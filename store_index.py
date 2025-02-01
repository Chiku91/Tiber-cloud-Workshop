from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("End-to-End-Medical-Chatbot-using-Llama-2\\data")
#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)
    text_chunks = text_split(extracted_data)
    
    return text_chunks

from transformers import AutoModel, AutoTokenizer
import torch

# Load pre-trained model and tokenizer from Hugging Face
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example text
sentences = ["Hello World"]
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Get embeddings
with torch.no_grad():
    embeddings = model(**inputs).last_hidden_state.mean(dim=1)

print(embeddings)
