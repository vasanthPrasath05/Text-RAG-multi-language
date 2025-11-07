# Text-RAG-Multi-Language

### Overview

Retrieval-Augmented Generation (RAG) is an AI technique that combines a retrieval system with a generative model to produce more accurate and context-aware responses. It works by first retrieving relevant information from an external knowledge base and then using that information to “ground” or guide the generative model’s response.
This approach helps reduce hallucinations (inaccurate information) and ensures that responses are factual and up to date.

## What is Multi-language RAG?
Multi-Language RAG enhances the RAG framework with multilingual capabilities. It allows the system to understand and generate responses in multiple languages, effectively breaking down language barriers in AI-driven applications.
 
### List of Libraries Used:
  1. streamlit – Used for building the web application UI.
  2. pypdf – Used to read and extract text from PDF files.
  3. google-generativeai – Used to call Google’s Large Language Model (LLM) APIs.
  4.langchain – Used for embedding models and managing the Vector Database (VectorDB).

## Steps to Run the Code:

### Step 1: Create a Virtual Environment.
  Create a Python virtual environment using one of the following commands:
    python -m venv your_venv_name or "python3 -m venv your_venv_name"(Linux/macos)

### Step 2: Activate the Virtual Environment.
 Activate your environment using:
   "Source your_venv_name/bin/activate" (Linux/macos)
   
### Step 3: Install Dependencies.
 Install all required libraries from the requirements.txt file:
  "pip install -r requirement.txt" or "pip3 install -r requirement.txt"

### Step 4: Setup Vector Database. 
  Create a folder named vectorDb in your current working directory.
  This folder will store the vector embeddings for retrieval.

### Step 5: Configure API Key.
  Replace the placeholder GOOGLE_API_KEY in the code with your actual API key.
  
Note: 
Run the Webpage.py by using the command:
  "streamlit run Webpage.py".

