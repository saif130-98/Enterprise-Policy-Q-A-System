# Enterprise Policy Q&A System

## Overview
This repository contains an AI-powered Retrieval-Augmented Generation (RAG) system designed to answer questions based on internal company policies. By utilizing dense vector embeddings and semantic search, the application accurately retrieves context-specific information from unstructured text and grounds the Large Language Model (LLM) responses in verified data, effectively mitigating AI hallucinations.

## Features
* **Interactive UI:** A user-friendly Streamlit web interface for seamless and intuitive querying[cite: 2].
* **Dense Passage Retrieval (DPR):** Utilizes Facebook's context and question encoders to map policy documents and user queries into a shared vector space.
* **High-Speed Vector Search:** Employs a FAISS (Facebook AI Similarity Search) Inner Product index for instant similarity matching[cite: 1, 2].
* **Hallucination-Resistant Generation:** Uses a generative language model (`gpt2`) with strict anti-looping (`no_repeat_ngram_size`) and penalty parameters to synthesize conversational answers derived exclusively from retrieved contexts[cite: 1, 2].
* **Resource Caching:** Leverages Streamlit's caching functionality to ensure large AI models and FAISS indices are loaded only once upon startup, making subsequent searches instantaneous[cite: 2].

## Repository Structure
* `RagPipeline.py`: The standalone Python script containing the foundational RAG architecture, text chunking logic, and backend testing functions.
* `RagApp.py`: The interactive Streamlit web application that wraps the pipeline into a graphical user interface, complete with confidence scores and source-viewing capabilities[cite: 2].

## Prerequisites
Ensure you have Python 3.10+ installed. It is highly recommended to use a virtual environment (like Conda) to manage dependencies.

### Required Libraries
Install the necessary packages using pip:
```bash
pip install streamlit torch numpy faiss-cpu transformers
