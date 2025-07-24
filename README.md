# ADOBE-Round1B

## **Loads three models:**

- **Bi-Encoder: For embedding text into vectors.**
- **Cross-Encoder: For fine-grained relevance scoring.**
- **Summarizer: For summarizing long paragraphs.**

## **processor.build_index_from_directory(PDF_DATA_PATH)**

- **Read all PDF files from the directory.**
- **Extract text blocks page-wise using fitz.**
- **For each text block:**
  - **If it’s bold and short → Treat as a heading.**
  - **Otherwise → Store as a chunk with metadata:**
- **Document name**
- **Page number**
- **Section title**
- **Generate embeddings for all chunks using the bi-encoder.**
- **Store these in a FAISS index for fast semantic retrieval.**

## **output_json = processor.process_query(QUERY, PERSONA, INPUT_DOCS)**

- **Semantic Search + Re-Ranking**
  - **Embed the query.**
  - **Search FAISS index for top k=150 similar chunks.**
  - **Rerank those using the cross-encoder to get top_k=50.**
- **Section Extraction**
  - **Get top 5 unique sections from the ranked chunks.**
  - **Each section is identified by its (document, section_title) pair.**
  - **Output includes section title, page number, etc.**
- **Summarization of Subsections**
  - **Use top 15 ranked chunks.**
  - **Perform community detection (clustering) using sentence similarity.**
  - **For each cluster of chunks:**
    - **Concatenate their texts.**
    - **Generate a summary using the summarizer.**
    - **Store associated doc names and page numbers.**
- **Final Output Assembly**
