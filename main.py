import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import json
import sys
from datetime import datetime
from collections import defaultdict

from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import pipeline

# --- Configuration ---
MODEL_PATHS = {
    'bi_encoder': './models/bi_encoder',
    'cross_encoder': './models/cross_encoder',
    'summarizer': './models/summarizer'
}
PDF_DATA_PATH = './data'

class DocumentProcessor:
    def __init__(self, model_paths):
        print("Initializing DocumentProcessor...")
        self.bi_encoder = SentenceTransformer(model_paths['bi_encoder'])
        self.cross_encoder = CrossEncoder(model_paths['cross_encoder'])
        self.summarizer = pipeline("summarization", model=model_paths['summarizer'], tokenizer=model_paths['summarizer'])
        
        self.chunks = []
        self.chunk_embeddings = None
        self.index = None
        print("DocumentProcessor initialized successfully.")

    def _chunk_pdf(self, doc_path):
        doc_name = os.path.basename(doc_path)
        print(f"  - Processing document: {doc_name}")
        document_chunks = []
        try:
            doc = fitz.open(doc_path)
            current_section_title = "Introduction"
            
            for page_num, page in enumerate(doc):
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        spans = block['lines'][0]['spans']
                        if not spans: continue
                        
                        is_heading = spans[0]['flags'] & 2**4
                        block_text = " ".join(
                            [span['text'] for line in block['lines'] for span in line['spans']]
                        ).strip()
                        
                        if len(block_text) < 30: # Increase min length to avoid noise
                            continue

                        if is_heading and len(block_text.split()) < 15: # Treat shorter bold text as headings
                            current_section_title = block_text
                        else:
                            chunk_data = {
                                'text': block_text.replace('\n', ' '),
                                'metadata': {
                                    'doc_name': doc_name,
                                    'page_number': page_num + 1,
                                    'section_title': current_section_title
                                }
                            }
                            document_chunks.append(chunk_data)
            return document_chunks
        except Exception as e:
            print(f"    - Error processing {doc_name}: {e}")
            return []

    def build_index_from_directory(self, pdf_directory):
        absolute_pdf_path = os.path.abspath(pdf_directory)
        print("\n--- Building Document Index ---")
        print(f"Attempting to read PDF files from: {absolute_pdf_path}")

        if not os.path.isdir(absolute_pdf_path):
            print(f"    > FATAL ERROR: Directory does not exist.")
            sys.exit(1)

        pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"    > FATAL ERROR: No PDF files found in the directory.")
            sys.exit(1)

        for pdf_path in pdf_files:
            self.chunks.extend(self._chunk_pdf(pdf_path))
            
        if not self.chunks:
            raise ValueError("No text chunks were extracted.")

        print(f"\nTotal chunks created: {len(self.chunks)}")
        print("Encoding chunks into vectors...")
        
        self.chunk_embeddings = self.bi_encoder.encode(
            [chunk['text'] for chunk in self.chunks],
            show_progress_bar=True,
            convert_to_tensor=False
        )

        print("Building FAISS index...")
        dimension = self.chunk_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.chunk_embeddings))
        print("--- Index built successfully ---")

    def _search_and_rerank(self, query, top_k_retrieval=150, top_k_rerank=50):
        # Widen the net even more
        query_embedding = self.bi_encoder.encode([query])
        _, top_k_indices = self.index.search(np.array(query_embedding), top_k_retrieval)
        
        candidate_indices = top_k_indices[0]
        candidate_chunks = [self.chunks[i] for i in candidate_indices]

        cross_encoder_inputs = [[query, chunk['text']] for chunk in candidate_chunks]
        rerank_scores = self.cross_encoder.predict(cross_encoder_inputs)

        for i, chunk in enumerate(candidate_chunks):
            chunk['rerank_score'] = rerank_scores[i]
            chunk['original_index'] = candidate_indices[i]
            
        ranked_chunks = sorted(candidate_chunks, key=lambda x: x['rerank_score'], reverse=True)
        
        return ranked_chunks[:top_k_rerank]

    # --- COMPLETELY NEW LOGIC FOR SECTIONS ---
    def _generate_main_sections(self, ranked_chunks):
        """
        NEW LOGIC: Generates 'extracted_sections' based on the SECTIONS of the
        TOP 5 most relevant CHUNKS. This is a "chunk-first" approach.
        """
        extracted_sections = []
        seen_sections = set()
        
        # Iterate through the highest-ranked chunks
        for chunk in ranked_chunks:
            # Create a unique identifier for a section within a document
            section_id = (chunk['metadata']['doc_name'], chunk['metadata']['section_title'])
            
            if section_id not in seen_sections:
                extracted_sections.append({
                    "document": chunk['metadata']['doc_name'],
                    "page_number": chunk['metadata']['page_number'],
                    "section_title": chunk['metadata']['section_title'],
                    "importance_rank": len(extracted_sections) + 1
                })
                seen_sections.add(section_id)
            
            # Stop once we have 5 unique sections
            if len(extracted_sections) >= 5:
                break
                
        return extracted_sections

    # --- COMPLETELY NEW LOGIC FOR SUBSECTIONS ---
    def _generate_dynamic_subsections(self, ranked_chunks):
        """
        NEW LOGIC: Creates summaries based on clusters formed ONLY from the
        absolute top-ranked chunks, ensuring high relevance.
        """
        # Take only the top ~15 chunks for high-quality clustering
        top_chunks_for_clustering = ranked_chunks[:15]
        if len(top_chunks_for_clustering) < 3:
            return []

        result_embeddings = np.array([self.chunk_embeddings[chunk['original_index']] for chunk in top_chunks_for_clustering])
        
        # Cluster these high-quality chunks
        clusters = util.community_detection(result_embeddings, min_community_size=3, threshold=0.65) # Adjusted threshold

        subsection_analysis = []
        for i, cluster in enumerate(clusters):
            # Ensure we don't re-summarize the same chunks
            if len(cluster) < 3: continue

            cluster_chunks = [top_chunks_for_clustering[j] for j in cluster]
            full_text = " ".join([chunk['text'] for chunk in cluster_chunks])
            
            if len(full_text.split()) > 40:
                summary = self.summarizer(full_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            else:
                summary = full_text

            docs = sorted(list(set([chunk['metadata']['doc_name'] for chunk in cluster_chunks])))
            pages = sorted(list(set([chunk['metadata']['page_number'] for chunk in cluster_chunks])))

            subsection_analysis.append({
                "document": docs[0] if len(docs) == 1 else docs,
                "refined_text": summary,
                "page_number": pages[0] if len(pages) == 1 else pages
            })
            
            if len(subsection_analysis) >= 5:
                break
                
        return subsection_analysis

    def process_query(self, query, persona, input_documents):
        print("\n--- Processing Query ---")

        print("Step 1/4: Searching for relevant information...")
        enhanced_query = f"{persona}: {query}"
        ranked_chunks = self._search_and_rerank(enhanced_query)

        # ✅ Save top 50 ranked chunks to chunk_data/chunks.json
        os.makedirs("chunk_data", exist_ok=True)
        chunks_to_save = [
            {
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "rerank_score": float(chunk["rerank_score"])  # Fix applied here
            }
            for chunk in ranked_chunks
        ]
        with open("chunk_data/chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_to_save, f, indent=2)
        print("Top 50 ranked chunks saved to chunk_data/chunks.json")


        print("Step 2/4: Identifying top sections from best chunks...")
        extracted_sections = self._generate_main_sections(ranked_chunks)

        print("Step 3/4: Summarizing most relevant content...")
        subsection_analysis = self._generate_dynamic_subsections(ranked_chunks)

        print("Step 4/4: Assembling final JSON output...")
        final_output = {
            "metadata": {
                "input_documents": input_documents,
                "persona": persona,
                "job_to_be_done": query,
                "processing_timestamp": datetime.utcnow().isoformat() + "Z"
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }

        print("--- Query Processed Successfully ---")
        return final_output



if __name__ == '__main__':
    processor = DocumentProcessor(MODEL_PATHS)
    processor.build_index_from_directory(PDF_DATA_PATH)

    # --- MORE DESCRIPTIVE QUERY FOR BETTER RESULTS ---
    QUERY = "Tell me about the key historical sites and artistic heritage of Marseille, specifically mentioning the Old Port, Notre-Dame de la Garde, Le Panier, and the MuCEM museum."
    PERSONA = "A history and art enthusiast preparing a detailed tour"
    
    INPUT_DOCS = [f for f in os.listdir(PDF_DATA_PATH) if f.endswith('.pdf')]
    output_json = processor.process_query(QUERY, PERSONA, INPUT_DOCS)

    print("\n\n--- FINAL JSON OUTPUT ---")
    print(json.dumps(output_json, indent=2))
    
    with open('final_output_corrected.json', 'w') as f:
        json.dump(output_json, f, indent=2)
    print("\nOutput also saved to final_output_corrected.json")