import streamlit as st
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

# --- 1. Page Configuration ---
st.set_page_config(page_title="Company Policies QA", layout="wide")
st.title("Company Policies Assistant 🤖")

# --- 2. Cache Models and Data ---
@st.cache_resource(show_spinner="Loading AI Models and Indexing Documents... (Takes a minute on first run)")
def load_system():
    def read_and_split_text(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                text = file.read()
            paragraphs = text.split('\n')
            return [para.strip() for para in paragraphs if len(para.strip()) > 0]
        except FileNotFoundError:
            st.error(f"Could not find '{filename}'. Please ensure it is in the same directory.")
            st.stop()

    paragraphs = read_and_split_text('companyPolicies.txt')
    
    # Load Context Encoder
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', use_safetensors=True)

    # Encode contexts
    embeddings = []
    with torch.no_grad():
        for text in paragraphs:
            inputs = context_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
            outputs = context_encoder(**inputs)
            embeddings.append(outputs.pooler_output)
    
    context_embeddings_np = torch.cat(embeddings).detach().numpy().astype('float32')

    # Build FAISS Index
    embedding_dim = 768
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(context_embeddings_np)

    # Load Question Encoder and Generator
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base', use_safetensors=True)
    
    generator_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    generator_model = AutoModelForCausalLM.from_pretrained('gpt2', use_safetensors=True)

    return paragraphs, index, question_tokenizer, question_encoder, generator_tokenizer, generator_model

# Load everything 
paragraphs, index, question_tokenizer, question_encoder, generator_tokenizer, generator_model = load_system()

# --- 3. Helper Functions ---
def search_relevant_contexts(question, k=5):
    question_inputs = question_tokenizer(question, return_tensors='pt')
    with torch.no_grad():
        question_embedding = question_encoder(**question_inputs).pooler_output.detach().numpy()
    
    D, I = index.search(question_embedding, k)
    return D, I

def generate_answer(question, contexts):
    context_str = "\n".join(contexts)
    input_text = f"Based on the following company policies:\n{context_str}\n\nQuestion: {question}\nAnswer:"
    
    inputs = generator_tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)

    summary_ids = generator_model.generate(
        inputs['input_ids'], 
        max_new_tokens=60, 
        length_penalty=1.0,
        num_beams=4, 
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=generator_tokenizer.eos_token_id
    )
    
    full_output = generator_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    answer_only = full_output.split("Answer:")[-1].strip()
    return answer_only

# --- 4. User Interface ---
st.markdown("Ask any question regarding internal company rules, and the AI will retrieve the relevant policy and generate an answer.")

user_question = st.text_input("Enter your question:", placeholder="e.g., What is the Smoking Policy?")

if st.button("Search") and user_question:
    with st.spinner("Searching policies and generating answer..."):
        
        # Retrieve
        D, I = search_relevant_contexts(user_question, k=5)
        retrieved_contexts = [paragraphs[idx] for idx in I[0]]
        
        # Generate
        answer = generate_answer(user_question, retrieved_contexts)
        
        # Display Answer
        st.subheader("Answer:")
        st.success(answer)
        
        # Display Sources
        with st.expander("View Retrieved Policy Sources"):
            for i, context in enumerate(retrieved_contexts):
                st.markdown(f"**Source {i+1}** (Confidence Score: {D[0][i]:.2f})")
                st.write(context)
                st.divider()

