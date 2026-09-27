# Enterprise Policy Q&A System

## Overview
This repository contains a fully local, production-ready Retrieval-Augmented Generation (RAG) system designed to answer questions based on 9 core internal company policies. By utilizing dense vector embeddings and semantic search, the application accurately retrieves context-specific information from unstructured text chunks and grounds the Large Language Model (LLM) responses in verified data. The system features a custom real-time Lexical Groundedness metric to actively detect and prevent AI hallucinations.

## Features
* **Instruction-Tuned Generation:** Built with Google's `flan-t5-base`, an open-source, sequence-to-sequence language model. Because this model is instruction-finetuned on a mixture of tasks, it strictly follows prompt directives to extract answers without auto-completing external, unverified information.
* **Dual-Encoder Dense Passage Retrieval (DPR):** Utilizes Facebook's context and question encoders (`dpr-ctx_encoder-single-nq-base`, `dpr-question_encoder-single-nq-base`) to map policy paragraphs and user queries into a shared 768-dimensional vector space.
* **High-Speed Vector Search:** Employs a FAISS Inner Product (`IndexFlatIP`) index for instantaneous similarity matching and top-k context retrieval.
* **Real-Time Hallucination Diagnostics:** Incorporates a custom-built Python evaluation metric that calculates the lexical overlap between generated answers and retrieved contexts to ensure strict faithfulness.
* **Interactive UI:** A streamlined Streamlit web interface (`RagApp.py`) that displays the generated answer, FAISS retrieval confidence, faithfulness scores, and expandable source document views.

## Technical Stack
* **Language:** Python
* **Frontend:** Streamlit
* **Machine Learning & NLP:** PyTorch, Hugging Face `transformers`
* **Vector Database:** FAISS
* **Models:** `google/flan-t5-base`, `facebook/dpr-ctx_encoder-single-nq-base`, `facebook/dpr-question_encoder-single-nq-base`

## Repository Structure
* `RagPipeline.py`: The standalone backend script containing the data chunking logic, FAISS indexing, FLAN-T5 generation parameters (`min_new_tokens`, `repetition_penalty`), and the `calculate_groundedness` evaluation function.
* `RagApp.py`: The interactive web application integrating the backend pipeline into a graphical user interface with `@st.cache_resource` for optimized model loading.
* `companyPolicies.txt`: The raw dataset containing the 9 core policy points separated by paragraphs.

## Local Setup & Deployment
This project is configured to run efficiently on local hardware by utilizing `use_safetensors=True` and strict `torch.no_grad()` memory management during inference.

1. **Install Dependencies:**
   Ensure Python 3.10+ is installed, then run:
   ```bash
   pip install streamlit torch numpy faiss-cpu transformers
