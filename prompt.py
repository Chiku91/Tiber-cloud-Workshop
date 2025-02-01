def generate_answer(context, question):
    prompt = f"""
    Answer the following question based on the context provided below. Provide a detailed, informative response. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}
    
    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    
    