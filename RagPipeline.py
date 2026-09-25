import wget
import torch
import numpy as np
import random
import faiss
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    AutoTokenizer, AutoModelForCausalLM
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
    # Added torch.no_grad() to prevent memory leaks during inference
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
index = faiss.IndexFlatIP(embedding_dim) # Inner Product is best for DPR
index.add(context_embeddings_np)

# 4. Load DPR Question Encoder
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base', use_safetensors=True)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

# 5. Load Generative Model
generator_tokenizer = AutoTokenizer.from_pretrained('gpt2')
generator_model = AutoModelForCausalLM.from_pretrained('gpt2', use_safetensors=True)

def search_relevant_contexts(question, question_tokenizer, question_encoder, index, k=5):
    question_inputs = question_tokenizer(question, return_tensors='pt')
    with torch.no_grad():
        question_embedding = question_encoder(**question_inputs).pooler_output.detach().numpy()
    
    # Search the index to retrieve top k relevant contexts
    D, I = index.search(question_embedding, k)
    return D, I

def generate_answer(question, contexts):
    # 1. Build a structured prompt so GPT-2 knows what to do
    context_str = "\n".join(contexts)
    input_text = f"Based on the following company policies:\n{context_str}\n\nQuestion: {question}\nAnswer:"
    
    inputs = generator_tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)

    # 2. Add anti-looping parameters (no_repeat_ngram_size)
    summary_ids = generator_model.generate(
        inputs['input_ids'], 
        max_new_tokens=60, 
        length_penalty=1.0,
        num_beams=4, 
        no_repeat_ngram_size=2, # This strictly prevents the model from looping
        early_stopping=True,
        pad_token_id=generator_tokenizer.eos_token_id
    )
    
    # 3. Decode and extract only the new generated text
    full_output = generator_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Split the output at "Answer:" and only return what the AI wrote
    answer_only = full_output.split("Answer:")[-1].strip()
    return answer_only

# 6. Test the Pipeline
question = 'tell me about Smoking Policy'
D, I = search_relevant_contexts(question, question_tokenizer, question_encoder, index, k=5)

print("\n--- Top 5 relevant contexts ---")
retrieved_contexts = []
for i, idx in enumerate(I[0]):
    context = paragraphs[idx]
    retrieved_contexts.append(context)
    print(f"{i+1} (Score: {D[0][i]:.2f}): {context}\n")

print("\n--- Generated Answer ---")
answer = generate_answer(question, retrieved_contexts)
print(answer)