# Hackathon_3_2025
AI Powered Document Search and Summarization System
# Objective:
Build an AI-driven tool to ingest documents, perform semantic search, and generate concise summaries of relevant sections.
## 1. Definining Scope & Document Types
* First, decide which document formats to support (PDF, Word, plain text).
* Ingestion workflow:

Upload → Text Extraction → Embedding → Vector Store → Query Interface → Summarization → Output

## 2. Implement Document Ingestion & Text Extraction
* First, set up file upload endpoints (e.g., in a simple Flask or Streamlit app).
* Then, use a library like PyPDF2 or python-docx to extract raw text from each document.

* *Using a gradio app the system handles uploading of multiple files using the UI*

## 3. Split & Preprocess Text
* First, segment the extracted text into manageable chunks (e.g., 200–500 tokens per chunk).
* Then, clean and normalize each chunk (remove headers, footers, whitespace).

* *For my system I decided to use `spaCy` preprocessing library to segment documents into sentences and then groups these sentences into chunks with a `CHUNK_SIZE` of 200 tokens and a `CHUNK_OVERLAP` of 50 tokens. This sentence-based chunking ensures semantic coherence within each chunk, which is crucial for both accurate embedding and meaningful summarization. `spaCy` was chosen for its robust NLP capabilities and local processing, suitable for CPU environments.*

* ## 4. Generate Embeddings
* First, load a sentence-transformer model (e.g., all-MiniLM-L6-v2).
* Then, compute embeddings for each text chunk.

## 5. Build & Populate the Vector Database
* First, initialize FAISS or Pinecone index with CPU-friendly settings.
* Then, insert your chunk embeddings and keep metadata (document ID, chunk index).

* * *We are using Pinecone to do both the embedding and the upload to the vector database so that we can have the advantage of offloading the embedding process to Pinecone - more resource efficient.
    Pinecone serves as the vector database, specifically utilizing its serverless `create_index_for_model` feature with the `llama-text-embed-v2` model. This approach offloads the embedding process to Pinecone, reducing local CPU load and simplifying the infrastructure. `llama-text-embed-v2` is a powerful model suitable for generating high-quality semantic embeddings. The metadata mapping (`document_id`, `chunk_index`) within Pinecone is critical for tracing retrieved information back to its source.*

## 6. Create the Search Interface
* First, build a query input form in your app.
* Then, on query submission: embed the query, search the vector store for top‑k chunks, and return the most relevant ones.
  
* *The `search_pinecone` function queries the Pinecone index using the `llama-text-embed-v2` model for embedding the user's query. It retrieves `top_k` (defaulting to 5) most relevant chunks based on semantic similarity. The search results include the original text, document ID, chunk index, and similarity score, which are then formatted for display.*

## 7. Summarize Retrieved Content
* First, load a summarization model (e.g., BART-base or t5-small).
* Then, pass the concatenated top‑k chunks into the summarizer to produce a concise summary.

* *A `t5-small` model from Hugging Face's `transformers` library is used for generating concise summaries. `t5-small` was specifically chosen for its smaller footprint and suitability for CPU-based inference, balancing summarization quality with computational efficiency. The `generate_summary` function preprocesses the concatenated retrieved chunks for the model and decodes the output.
The `rag_pipeline` function orchestrates the entire process, from query embedding and retrieval to summarization. It ensures a seamless flow of information and provides both the raw retrieved content and the generated summary.*

## 8. Evaluate Search & Summarization
* First, assemble a test set of queries with ground‑truth relevant passages and summaries.
* Then, compute:
    * Search: precision@k, recall@k
    * Summarization: BLEU, ROUGE, perplexity

 * *Ground Truth datasets were generated for the three test files so we can implement evaluation using rouge score for retrieval and summarization.
Everything was integrated using the gradio interface providing a common interface for document uploading, tokenization and embedding, query input for embedding and retrieving results and evaluation of the system for retrieval and summarization using the rouge scores* 
