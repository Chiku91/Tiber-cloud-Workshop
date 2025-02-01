from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')

# Load model and tokenizer from HuggingFace for GPT-2
model_name = "gpt2"
gpt2_model = AutoModelForCausalLM.from_pretrained(model_name)
gpt2_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use Sentence-Transformer for embeddings
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Example text (replace with your actual text chunks)
text_chunks = [
    {"page_content": "Allergies are the body's immune response to substances."},
    {"page_content": "A headache can occur from various conditions, including allergies."},
    {"page_content": "Fever is a common symptom of infection."},
    {"page_content": "Skin rashes may appear when someone has an allergy."},
    {"page_content": "Treatment for allergies often involves antihistamines."}
]

# Generate embeddings for the text chunks using SentenceTransformer
def generate_embeddings(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        sentence = chunk['page_content']
        embedding = embedding_model.encode(sentence)
        embeddings.append(embedding)
    return np.array(embeddings)

# Perform similarity search
def similarity_search(query, embeddings, text_chunks, top_k=3):
    query_embedding = embedding_model.encode(query)
    similarities = cosine_similarity([query_embedding], embeddings)
    top_k_indices = similarities.argsort()[0][-top_k:][::-1]
    results = [text_chunks[i] for i in top_k_indices]
    
    return results

def generate_answer(context, question):
    prompt = f"""
    Answer the following question based on the context provided below. Provide a detailed, informative response. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}
    
    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    
    # Prepare input for GPT-2
    inputs = gpt2_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate a longer answer using GPT-2 by increasing max_length
    outputs = gpt2_model.generate(
        inputs["input_ids"],
        max_length=350,  # Increase this to a larger value for longer responses
        temperature=1.0,  # Slightly higher for more variety
        top_p=1.0,  # Increased to allow for more creativity
        num_return_sequences=1,  # Only generate one response, you can adjust this if needed
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    # Decode the answer
    answer = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove everything before the "Helpful answer:" part
    helpful_answer = answer.split("Helpful answer:")[-1].strip()

    return helpful_answer


# Main query function to get the helpful answer
def get_helpful_answer(query):
    # Generate embeddings for text chunks (only once)
    embeddings = generate_embeddings(text_chunks)
    
    # Perform similarity search to get relevant context based on the query
    results = similarity_search(query, embeddings, text_chunks)
    
    # Combine the relevant context for the prompt
    context = "\n".join([result['page_content'] for result in results])
    
    try:
        response = generate_answer(context, query)  # Generate the answer using GPT-2
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, there was an error processing your request."

# Flask route to render the template
@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    if request.method == "POST":
        user_input = request.form["query"]
        response = get_helpful_answer(user_input)  # Get response based on user input
    return render_template('chat.html', response=response)


if __name__ == '__main__':
    app.run(debug=True)
