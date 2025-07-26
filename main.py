import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import json
import sys
from datetime import datetime, timezone
from collections import defaultdict, Counter
import statistics
import re

from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import pipeline

from utils.local_model import LocalHeadingModel

# --- Configuration ---
MODEL_PATHS = {
    'bi_encoder': './models/bi_encoder',
    'cross_encoder': './models/cross_encoder',
    'summarizer': './models/summarizer',
    'labler': './models/labler'
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
        section_titles = []

        try:
            doc = fitz.open(doc_path)

            all_font_sizes = []
            for page in doc:
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        for span in line["spans"]:
                            all_font_sizes.append(span["size"])

            base_font_size = statistics.median(all_font_sizes) if all_font_sizes else 10
            heading_threshold = base_font_size * 1.05
            current_section_title = "Introduction"

            for page_num, page in enumerate(doc):
                page_width, page_height = page.rect.width, page.rect.height
                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if "lines" not in block:
                        continue

                    lines = []
                    max_font_size = 0
                    is_bold = False
                    is_italic = False
                    is_all_caps = True
                    has_numbers = False
                    bbox = block.get("bbox", [0, 0, 0, 0])

                    for line in block["lines"]:
                        line_text = "".join([span["text"] for span in line["spans"]]).strip()
                        if not line_text:
                            continue
                        lines.append(line_text)

                        for span in line["spans"]:
                            font = span["font"].lower()
                            size = span["size"]
                            max_font_size = max(max_font_size, size)

                            if "bold" in font or "black" in font or "heavy" in font:
                                is_bold = True
                            if "italic" in font or "oblique" in font:
                                is_italic = True
                            if any(c.islower() for c in span["text"]):
                                is_all_caps = False
                            if any(c.isdigit() for c in span["text"]):
                                has_numbers = True

                    if not lines:
                        continue

                    block_text = " ".join(lines).strip()
                    word_count = len(block_text.split())

                    starts_with_bullet = bool(re.match(r"^(\*|•|-|–|—|\d+[\.\)]|[a-zA-Z][\.\)])", block_text))
                    is_short = word_count <= 15
                    is_title_case = block_text.istitle() or block_text.isupper()
                    ends_with_colon = block_text.endswith(':')

                    heading_patterns = [
                        r"^\d+\.?\s+[A-Z]",
                        r"^[A-Z][a-z]+\s+\d+",
                        r"^[A-Z]{2,}",
                        r"^\d+\.\d+",
                        r"^[IVX]+\.?",
                    ]
                    matches_heading_pattern = any(re.match(pattern, block_text) for pattern in heading_patterns)
                    is_left_aligned = bbox[0] < 100

                    heading_score = 0
                    if max_font_size >= heading_threshold:
                        heading_score += 2
                    elif max_font_size >= base_font_size:
                        heading_score += 1
                    if is_bold:
                        heading_score += 3
                    if is_all_caps and word_count <= 8:
                        heading_score += 2
                    if is_title_case:
                        heading_score += 1
                    if is_italic:
                        heading_score += 1
                    if is_short:
                        heading_score += 2
                    if matches_heading_pattern:
                        heading_score += 3
                    if ends_with_colon:
                        heading_score += 1
                    if is_left_aligned:
                        heading_score += 1
                    if starts_with_bullet:
                        heading_score -= 10
                    if word_count > 20:
                        heading_score -= 2
                    if block_text.endswith('.') and not matches_heading_pattern:
                        heading_score -= 1

                    is_heading = heading_score >= 4

                    x0, y0, x1, y1 = bbox

                    if is_heading and not starts_with_bullet:
                        if (is_all_caps or max_font_size >= base_font_size * 1.3 or 
                            re.match(r"^(chapter|section|part)\s+\d+", block_text.lower())):
                            heading_level = "H1"
                        elif (is_bold and (has_numbers or matches_heading_pattern)):
                            heading_level = "H2"
                        else:
                            heading_level = "H3"

                        current_section_title = block_text

                        section_titles.append({
                            "text": block_text,
                            "bbox": {
                                "x0": x0,
                                "y0": y0,
                                "x1": x1,
                                "y1": y1
                            },
                            "font_size": round(max_font_size, 2),
                            "is_bold": is_bold,
                            "is_italic": is_italic,
                            "page_number": page_num + 1,
                            "line_position": y0,
                            "width": x1 - x0,
                            "height": y1 - y0,
                            "relative_x": round(x0 / page_width, 6),
                            "relative_y": round(y0 / page_height, 6),
                            "page_width": page_width,
                            "page_height": page_height,
                            "source": "digital"
                        })
                        continue

                    chunk_data = {
                        "text": block_text.replace('\n', ' '),
                        "bbox": {
                            "x0": x0,
                            "y0": y0,
                            "x1": x1,
                            "y1": y1
                        },
                        "font_size": round(max_font_size, 2),
                        "is_bold": is_bold,
                        "is_italic": is_italic,
                        "page_number": page_num + 1,
                        "line_position": y0,
                        "width": x1 - x0,
                        "height": y1 - y0,
                        "relative_x": x0 / page_width,
                        "relative_y": y0 / page_height,
                        "page_width": page_width,
                        "page_height": page_height,
                        "source": "digital",
                        "metadata": {
                            "doc_name": doc_name,
                            "page_number": page_num + 1,
                            "section_title": current_section_title,
                            "font_size": max_font_size,
                            "word_count": word_count
                        }
                    }
                    document_chunks.append(chunk_data)

        except Exception as e:
            print(f"[ERROR] Failed to chunk {doc_name}: {e}")
            return []

        # ✨ NEW: Predict heading labels using local model
        if section_titles:
            try:
                model = LocalHeadingModel()
                if model.load_model():
                    predictions = model.predict(section_titles)
                    for st, label in zip(section_titles, predictions):
                        st["predicted_label"] = label
                else:
                    print("Warning: Could not load heading model to label section_titles.")
            except Exception as e:
                print(f"Warning: Failed to label section_titles: {e}")

        # Save content chunks
        os.makedirs("chunk_data", exist_ok=True)
        output_path = os.path.join("chunk_data", f"{os.path.splitext(doc_name)[0]}_chunks.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(document_chunks, f, indent=2, ensure_ascii=False)

        # Save section titles (now with predicted_label)
        os.makedirs("section_titles", exist_ok=True)
        title_path = os.path.join("section_titles", f"{os.path.splitext(doc_name)[0]}_titles.json")
        with open(title_path, "w", encoding="utf-8") as f:
            json.dump(section_titles, f, indent=2, ensure_ascii=False)

        return document_chunks



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

    def _extract_key_terms(self, text, max_terms=10):
        """Extract key terms from text for enhanced matching."""
        # Basic implementation - enhance with proper NLP processing
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                     'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        # Extract words (alphanumeric, 3+ chars)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered_words = [w for w in words if w not in stop_words]
        
        # Return most frequent terms
        word_counts = Counter(filtered_words)
        return [word for word, _ in word_counts.most_common(max_terms)]

    def _construct_enhanced_query(self, query, persona, job_description=None):
        """Construct enhanced query incorporating persona and job context."""
        enhanced_parts = [query]
        
        if persona:
            # Extract key persona attributes
            persona_keywords = self._extract_key_terms(persona, max_terms=5)
            enhanced_parts.extend(persona_keywords)
        
        if job_description:
            # Extract key job requirements
            job_keywords = self._extract_key_terms(job_description, max_terms=5)
            enhanced_parts.extend(job_keywords)
        
        return " ".join(enhanced_parts)

    def _prepare_rerank_inputs(self, chunks, query, persona, job_description=None):
        """Prepare optimized inputs for cross-encoder reranking."""
        rerank_inputs = []
        
        # Create context-aware query for reranking
        context_query = query
        if persona and job_description:
            context_query = f"{query} [Persona: {persona[:100]}...] [Job: {job_description[:100]}...]"
        elif persona:
            context_query = f"{query} [Persona: {persona[:150]}...]"
        elif job_description:
            context_query = f"{query} [Job: {job_description[:150]}...]"
        
        for chunk in chunks:
            rerank_inputs.append([context_query, chunk['text']])
        
        return rerank_inputs

    def _calculate_persona_match(self, text, persona):
        """Calculate how well the text matches the given persona."""
        persona_terms = set(self._extract_key_terms(persona, max_terms=10))
        text_terms = set(self._extract_key_terms(text.lower(), max_terms=20))
        
        if not persona_terms:
            return 0
        
        overlap = len(persona_terms.intersection(text_terms))
        return overlap / len(persona_terms)

    def _calculate_job_relevance(self, text, job_description):
        """Calculate relevance to job requirements."""
        job_terms = set(self._extract_key_terms(job_description, max_terms=15))
        text_terms = set(self._extract_key_terms(text.lower(), max_terms=20))
        
        if not job_terms:
            return 0
        
        overlap = len(job_terms.intersection(text_terms))
        jaccard_score = overlap / len(job_terms.union(text_terms)) if job_terms.union(text_terms) else 0
        
        return jaccard_score

    def _search_and_rerank(self, query, persona=None, job_description=None, 
                           top_k_retrieval=150, top_k_rerank=50, 
                           persona_weight= 0.8, job_weight= 1):
        """
        Enhanced search and rerank with persona and job description matching.
        
        Args:
            query: Main search query
            persona: Target persona description for matching
            job_description: Job requirements for relevance scoring
            top_k_retrieval: Initial retrieval count
            top_k_rerank: Final reranked results count
            persona_weight: Weight for persona matching score
            job_weight: Weight for job description matching score
        """
        
        # 1. Enhanced query construction
        enhanced_query = self._construct_enhanced_query(query, persona, job_description)
        
        # 2. Multi-vector retrieval
        query_embedding = self.bi_encoder.encode([enhanced_query])
        similarities, top_k_indices = self.index.search(np.array(query_embedding), top_k_retrieval)
        
        candidate_indices = top_k_indices[0]
        candidate_chunks = []
        
        # 3. Pre-filter based on similarity threshold
        similarity_threshold = np.percentile(similarities[0], 25)  # Keep top 75%
        
        for i, idx in enumerate(candidate_indices):
            if similarities[0][i] >= similarity_threshold:
                chunk = self.chunks[idx].copy()
                chunk['retrieval_score'] = float(similarities[0][i])
                chunk['original_index'] = idx
                candidate_chunks.append(chunk)
        
        if not candidate_chunks:
            return []
        
        # 4. Batch reranking with context
        rerank_inputs = self._prepare_rerank_inputs(candidate_chunks, query, persona, job_description)
        rerank_scores = self.cross_encoder.predict(rerank_inputs)
        
        # 5. Multi-criteria scoring
        for i, chunk in enumerate(candidate_chunks):
            # Base rerank score
            chunk['rerank_score'] = float(rerank_scores[i])
            
            # Persona matching score
            persona_score = 0
            if persona:
                persona_score = self._calculate_persona_match(chunk['text'], persona)
            
            # Job relevance score
            job_score = 0
            if job_description:
                job_score = self._calculate_job_relevance(chunk['text'], job_description)
            
            # Combined score
            chunk['final_score'] = (
                chunk['rerank_score'] + 
                persona_weight * persona_score + 
                job_weight * job_score
            )
            
            # Store component scores for analysis
            chunk['persona_score'] = persona_score
            chunk['job_score'] = job_score
        
        # 6. Final ranking and return
        ranked_chunks = sorted(candidate_chunks, key=lambda x: x['final_score'], reverse=True)
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
            
            extracted_sections.append({
                "document": chunk['metadata']['doc_name'],
                "page_number": chunk['metadata']['page_number'],
                "section_title": chunk['metadata']['section_title'],
                "importance_rank": len(extracted_sections) + 1,
                "final_score": chunk['final_score'],
                "persona_alignment": chunk.get('persona_score', 0),
                "job_relevance": chunk.get('job_score', 0)
            })
            if section_id not in seen_sections:
                seen_sections.add(section_id)
            
            # Stop once we have 5 unique sections
            if len(seen_sections) >= 5:
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
            avg_score = sum([chunk['final_score'] for chunk in cluster_chunks]) / len(cluster_chunks)

            subsection_analysis.append({
                "document": docs[0] if len(docs) == 1 else docs,
                "refined_text": summary,
                "page_number": pages[0] if len(pages) == 1 else pages,
                "cluster_score": float(avg_score),
                "chunk_count": len(cluster_chunks)
            })
            
            if len(subsection_analysis) >= 5:
                break
                
        return subsection_analysis

    def process_query(self, query, persona, input_documents, job_description=None, 
                     persona_weight=0.3, job_weight=0.2):
        print("\n--- Processing Query ---")

        print("Step 1/4: Searching for relevant information with enhanced ranking...")
        
        # Use enhanced search with persona and job description
        ranked_chunks = self._search_and_rerank(
            query=query, 
            persona=persona, 
            job_description=job_description,
            persona_weight=persona_weight,
            job_weight=job_weight
        )

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
                "processing_timestamp": datetime.now(timezone.utc).isoformat()
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }

        print("--- Query Processed Successfully ---")
        return final_output

def main():
    """
    Main function to process PDFs for travel planning.
    You can modify the query below or pass it as a command line argument.
    """
    processor = DocumentProcessor(MODEL_PATHS)
    processor.build_index_from_directory(PDF_DATA_PATH)

    # --- Travel Planning Configuration ---
    PERSONA = "Travel Planner"
    QUERY = "Plan a trip of 4 days for a group of 10 college friends"
    
    INPUT_DOCS = [f for f in os.listdir(PDF_DATA_PATH) if f.endswith('.pdf')]
    
    # Process with travel planning focused parameters
    output_json = processor.process_query(
        query=QUERY, 
        persona=PERSONA, 
        input_documents=INPUT_DOCS,
        persona_weight=0.4,  # Higher weight for persona matching
        job_weight=0.3       # Moderate weight for job relevance
    )
    
    with open('travel_plan_output.json', 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    print("\nTravel planning output saved to travel_plan_output.json")

if __name__ == '__main__':
    main()