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

* *Using a gradio app the system handles uploading of multiple files using the UI.
  Both PdPDF and python-docx were used to allow ingestion of PDFs and word docx documents*

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

## 9. Optimize for CPU & Scale
* First, ensure batch embedding and summarization with small batch sizes (2–4).
* Then, disable large FAISS features (e.g., IVF) and subsample documents if needed.

* *The notebook is already optimized for CPU with the right embeddings and summarization*

## 10. (Bonus) Automate New Document Processing
* First, set up a file-watcher or message queue to detect new uploads.
* Then, trigger your ingestion → embedding → indexing pipeline automatically.

* *The gradio app allows for ingestion of new documents but the scoring is only based on the three basic files used at the start because it is not possible to have ground truth to score for each new document*

## 11. Document & Reflect
* First, write a summary of your system architecture and component choices.
* Then, analyze search accuracy, summarization quality, and CPU performance trade‑offs.

* *The system architecture was analyzed at the top*

### Analysis of Search Accuracy, Summarization Quality, and CPU Performance Trade-offs

**Search Accuracy (Retrieval)**:

*   **Strengths**: The RAG architecture, powered by `llama-text-embed-v2` embeddings in Pinecone, is expected to provide strong semantic search capabilities. The chunking strategy, which preserves sentence integrity and introduces overlap, contributes to retrieving semantically complete and relevant passages. The `calculate_retrieval_hit` metric, which checks for the presence of ground-truth passages in retrieved text, is a direct measure of this.
*   **Trade-offs**: The accuracy is highly dependent on the quality of the embedding model and the chunking strategy. While `llama-text-embed-v2` is robust, extremely nuanced queries or documents with very subtle semantic distinctions might still pose challenges. The `top_k` parameter influences recall; a higher `top_k` increases the chances of retrieving all relevant information but also introduces more noise.
*   **CPU Performance**: Search operations, particularly the embedding of the query, are handled efficiently by Pinecone's integrated embedding, which significantly offloads the CPU. Local operations for orchestrating the search are minimal.

**Summarization Quality**:

*   **Strengths**: The `t5-small` model, while compact, offers decent summarization capabilities for general text. Its ability to generate coherent and concise summaries from the concatenated retrieved chunks is a key strength. The `ROUGE` scores (calculated against ground-truth summaries) provide a quantitative measure of content overlap and linguistic quality.
*   **Trade-offs**: `t5-small` is a trade-off between quality and performance. Larger models (e.g., `t5-large`, `BART`) would likely produce higher quality, more nuanced summaries but would require significantly more computational resources (GPU, memory) and incur higher latency. For a CPU-friendly, local solution, `t5-small` strikes a reasonable balance. Summaries might occasionally miss very specific details if they are not prominent in the retrieved chunks or if the model's capacity is limited.
*   **CPU Performance**: Summarization is the most computationally intensive step on the local CPU. Although `t5-small` is chosen for efficiency, generating summaries, especially for longer concatenated texts, can still introduce noticeable latency on a CPU-only setup. Batching (not explicitly implemented for individual summaries but inherent in how the model processes text) and model quantization are further optimizations for CPU performance, though `t5-small` already benefits from its smaller size.

**Overall CPU Performance**:

*   **Efficiency**: The design heavily leans on Pinecone for the most computationally demanding parts (embedding all document chunks and query embedding), making the local component CPU-friendly. The choice of `spaCy` (for chunking) and `t5-small` (for summarization) further prioritizes CPU performance over maximal accuracy or state-of-the-art model size.
*   **Bottlenecks**: The primary local CPU bottleneck will be the summarization step, particularly when `rag_pipeline` is called with many or very long retrieved chunks, as `t5-small` still performs inference on the CPU. The `preprocess` function (spaCy tokenization) is generally fast but could become a factor with extremely large single documents during ingestion.
*   **Scalability**: For scaling to a very large number of documents or very high query throughput, further optimizations would be needed, such as deploying the summarization model on a dedicated inference server or using a GPU-enabled environment. However, for a prototyping or small-to-medium scale application on a CPU, the current architecture is a pragmatic choice.
