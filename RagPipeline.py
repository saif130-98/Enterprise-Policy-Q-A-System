import wget
import torch
import numpy as np
import random
import faiss
import string
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
from transformers import logging as hf_logging

# Silence the harmless "UNEXPECTED weights" warnings from DPR
hf_logging.set_verbosity_error()

def read_and_split_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    # Split the text into paragraphs 
    paragraphs = text.split('\n')
    # Filter out empty paragraphs
    paragraphs = [para.strip() for para in paragraphs if len(para.strip()) > 0]
    return paragraphs

# 1. Load Data
paragraphs = read_and_split_text('companyPolicies.txt')
random.seed(42)
random.shuffle(paragraphs)

# 2. Load DPR Context Encoder
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', use_safetensors=True)

def encode_contexts(text_list):
    embeddings = []
    with torch.no_grad():
        for text in text_list:
            inputs = context_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
            outputs = context_encoder(**inputs)
            embeddings.append(outputs.pooler_output)
    return torch.cat(embeddings).detach().numpy()

# 3. Create Embeddings and FAISS Index
print("Encoding contexts... this may take a moment.")
context_embeddings = encode_contexts(paragraphs)
context_embeddings_np = np.array(context_embeddings).astype('float32')

# INITIALIZE AND POPULATE FAISS INDEX 
embedding_dim = 768
index = faiss.IndexFlatIP(embedding_dim) 
index.add(context_embeddings_np)

# 4. Load DPR Question Encoder
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base', use_safetensors=True)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

# 5. Load Generative Model (Swapped to FLAN-T5)
generator_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
generator_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base', use_safetensors=True)

def search_relevant_contexts(question, question_tokenizer, question_encoder, index, k=5):
    question_inputs = question_tokenizer(question, return_tensors='pt')
    with torch.no_grad():
        question_embedding = question_encoder(**question_inputs).pooler_output.detach().numpy()
    
    D, I = index.search(question_embedding, k)
    return D, I

def generate_answer(question, contexts):
    context_str = "\n".join(contexts)
    
    # 1. Update the prompt to explicitly demand a comprehensive explanation
    input_text = f"Based on the following company policies, provide a detailed and complete explanation to answer the user's question.\n\nContext: {context_str}\n\nQuestion: {question}\n\nDetailed Answer:"
    
    inputs = generator_tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)

    # 2. Add generation constraints to force longer, more detailed output
    summary_ids = generator_model.generate(
        inputs['input_ids'], 
        min_new_tokens=30,       # Forces the model to write at least 30 tokens
        max_new_tokens=150,      # Gives it more room to expand
        length_penalty=2.0,      # Heavily encourages longer text sequences
        repetition_penalty=1.2,  # Prevents it from repeating the same phrase
        early_stopping=True
    )
    
    return generator_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 6. Local Evaluation System
def calculate_groundedness(answer, contexts):
    """
    Evaluates Faithfulness/Hallucination by checking what percentage of 
    the generated content words actually exist in the retrieved policy chunks.
    """
    # Clean and tokenize contexts
    context_text = " ".join(contexts).lower().translate(str.maketrans('', '', string.punctuation))
    context_words = set(context_text.split())
    
    # Clean and tokenize answer
    ans_text = answer.lower().translate(str.maketrans('', '', string.punctuation))
    ans_words = ans_text.split()
    
    # Ignore common grammatical stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'and', 'or', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'as', 'of', 'this', 'that', 'it', 'be', 'from', 'has', 'have', 'will', 'not', 'no'}
    ans_content_words = [w for w in ans_words if w not in stop_words]
    
    if not ans_content_words:
        return 0.0
        
    # Calculate word overlap
    supported_words = [w for w in ans_content_words if w in context_words]
    return len(supported_words) / len(ans_content_words)

# 7. Test and Evaluate the Pipeline
question = 'tell me about Smoking Policy'
D, I = search_relevant_contexts(question, question_tokenizer, question_encoder, index, k=5)

retrieved_contexts = [paragraphs[idx] for idx in I[0]]
answer = generate_answer(question, retrieved_contexts)

# Calculate Metrics
avg_retrieval_score = float(np.mean(D[0]))
groundedness_score = calculate_groundedness(answer, retrieved_contexts)

# Output Results
print(f"\n--- Evaluation Results ---")
print(f"Question: '{question}'")
print(f"Average FAISS Retrieval Confidence: {avg_retrieval_score:.2f}")
print(f"Lexical Groundedness (Faithfulness): {groundedness_score * 100:.1f}%")
print(f"Verdict: " + ("Highly Grounded ✅" if groundedness_score > 0.75 else "Potential Hallucination Detected ⚠️"))

print("\n--- Generated Answer ---")
print(answer)
