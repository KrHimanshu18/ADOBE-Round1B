import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import json
import sys
import argparse
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
    'labeler': './models/labeler'
}
PDF_DATA_PATH = './data'

class DocumentProcessor:
    def __init__(self, model_paths):
        print("Initializing DocumentProcessor...")
        self.bi_encoder = SentenceTransformer(model_paths['bi_encoder'])
        self.cross_encoder = CrossEncoder(model_paths['cross_encoder'])
        self.summarizer = pipeline("summarization", model=model_paths['summarizer'], tokenizer=model_paths['summarizer'])
        
        # Initialize the labeler model
        # Fixed:
        self.labeler_model = LocalHeadingModel(model_dir=model_paths['labeler'])
        if not self.labeler_model.load_model():
            print("Warning: Could not load labeler model.")
        
        self.chunks = []
        self.labeled_chunks = []  # Store chunks with labels
        self.chunk_embeddings = None
        self.index = None
        print("DocumentProcessor initialized successfully.")

    def _chunk_pdf(self, doc_path):
        doc_name = os.path.basename(doc_path)
        print(f"  - Processing document: {doc_name}")
        document_chunks = []

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

                    if not lines:
                        continue

                    block_text = " ".join(lines).strip()
                    word_count = len(block_text.split())

                    x0, y0, x1, y1 = bbox

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
                        "page_number": page_num,
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
                            "page_number": page_num,
                            "font_size": max_font_size,
                            "word_count": word_count
                        }
                    }
                    document_chunks.append(chunk_data)

        except Exception as e:
            print(f"[ERROR] Failed to chunk {doc_name}: {e}")
            return []

        return document_chunks

    def _label_chunks(self, chunks):
        """Label all chunks using the labeler model."""
        print("Labeling chunks...")
        labeled_chunks = []
        
        if not chunks:
            return labeled_chunks

        try:
            # Prepare data for labeling (similar format as section_titles)
            chunks_for_labeling = []
            for chunk in chunks:
                chunk_for_label = {
                    "text": chunk["text"],
                    "doc_name": chunk["metadata"]["doc_name"],
                    "bbox": chunk["bbox"],
                    "font_size": chunk["font_size"],
                    "is_bold": chunk["is_bold"],
                    "is_italic": chunk["is_italic"],
                    "page_number": chunk["page_number"],
                    "line_position": chunk["line_position"],
                    "width": chunk["width"],
                    "height": chunk["height"],
                    "relative_x": chunk["relative_x"],
                    "relative_y": chunk["relative_y"],
                    "page_width": chunk["page_width"],
                    "page_height": chunk["page_height"],
                    "source": chunk["source"]
                }
                chunks_for_labeling.append(chunk_for_label)

            # Get predictions from labeler model
            predictions = self.labeler_model.predict(chunks_for_labeling)
            
            # Add labels to chunks
            for chunk, label in zip(chunks, predictions):
                labeled_chunk = chunk.copy()
                labeled_chunk["predicted_label"] = label
                labeled_chunks.append(labeled_chunk)
                
        except Exception as e:
            print(f"Warning: Failed to label chunks: {e}")
            # If labeling fails, assign 'NONE' to all chunks
            for chunk in chunks:
                labeled_chunk = chunk.copy()
                labeled_chunk["predicted_label"] = "NONE"
                labeled_chunks.append(labeled_chunk)

        return labeled_chunks

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

        # Step 2: Extract chunks
        all_chunks = []
        for pdf_path in pdf_files:
            chunks = self._chunk_pdf(pdf_path)
            all_chunks.extend(chunks)
            
        if not all_chunks:
            raise ValueError("No text chunks were extracted.")

        print(f"Total chunks extracted: {len(all_chunks)}")

        # Step 3: Label all chunks
        self.labeled_chunks = self._label_chunks(all_chunks)
        self.chunks = self.labeled_chunks  # Use labeled chunks as main chunks
        
        # Save labeled chunks to chunk_data folder
        os.makedirs("chunk_data", exist_ok=True)
        labeled_chunks_path = os.path.join("chunk_data", "all_labeled_chunks.json")
        with open(labeled_chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.labeled_chunks, f, indent=2, ensure_ascii=False)
        print(f"Labeled chunks saved to: {labeled_chunks_path}")

        # Step 4: Encode all chunks
        print("Encoding chunks into vectors...")
        self.chunk_embeddings = self.bi_encoder.encode(
            [chunk['text'] for chunk in self.chunks],
            show_progress_bar=True,
            convert_to_tensor=False
        )

        # Step 5: Build FAISS index
        print("Building FAISS index...")
        dimension = self.chunk_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.chunk_embeddings))
        print("--- Index built successfully ---")

    def _extract_key_terms(self, text, max_terms=10):
        """Extract key terms from text for enhanced matching."""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                     'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered_words = [w for w in words if w not in stop_words]
        
        word_counts = Counter(filtered_words)
        return [word for word, _ in word_counts.most_common(max_terms)]

    def _construct_enhanced_query(self, query, persona, job_description=None):
        """Construct enhanced query incorporating persona and job context."""
        enhanced_parts = [query]
        
        if persona:
            persona_keywords = self._extract_key_terms(persona, max_terms=5)
            enhanced_parts.extend(persona_keywords)
        
        if job_description:
            job_keywords = self._extract_key_terms(job_description, max_terms=5)
            enhanced_parts.extend(job_keywords)
        
        return " ".join(enhanced_parts)

    def _prepare_rerank_inputs(self, chunks, query, persona, job_description=None):
        """Prepare optimized inputs for cross-encoder reranking."""
        rerank_inputs = []
        
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

    def _calculate_word_count_penalty(self, text, ideal_word_count=8, penalty_factor=0.1):
        """
        Calculate a penalty/bonus based on word count to favor shorter texts.
        
        Args:
            text: The text to analyze
            ideal_word_count: The ideal number of words (gets no penalty)
            penalty_factor: How much to penalize for each word above ideal
            
        Returns:
            A score between -1 and 1, where shorter texts get higher scores
        """
        word_count = len(text.split())
        
        if word_count <= ideal_word_count:
            # Bonus for being at or under ideal length
            return min(1.0, (ideal_word_count - word_count + 1) * 0.1)
        else:
            # Penalty for being over ideal length
            excess_words = word_count - ideal_word_count
            penalty = excess_words * penalty_factor
            return max(-1.0, -penalty)

    def _search_and_rerank(self, query, persona=None, job_description=None, 
                           top_k_retrieval=150, top_k_rerank=50, 
                           persona_weight=0.8, job_weight=1):
        """
        Enhanced search and rerank with persona and job description matching.
        """
        # Step 6: Enhanced search and reranking
        enhanced_query = self._construct_enhanced_query(query, persona, job_description)
        
        query_embedding = self.bi_encoder.encode([enhanced_query])
        similarities, top_k_indices = self.index.search(np.array(query_embedding), top_k_retrieval)
        
        candidate_indices = top_k_indices[0]
        candidate_chunks = []
        
        similarity_threshold = np.percentile(similarities[0], 25)
        
        for i, idx in enumerate(candidate_indices):
            if similarities[0][i] >= similarity_threshold:
                chunk = self.chunks[idx].copy()
                chunk['retrieval_score'] = float(similarities[0][i])
                chunk['original_index'] = idx
                candidate_chunks.append(chunk)
        
        if not candidate_chunks:
            return []
        
        rerank_inputs = self._prepare_rerank_inputs(candidate_chunks, query, persona, job_description)
        rerank_scores = self.cross_encoder.predict(rerank_inputs)
        
        for i, chunk in enumerate(candidate_chunks):
            chunk['rerank_score'] = float(rerank_scores[i])
            
            persona_score = 0
            if persona:
                persona_score = self._calculate_persona_match(chunk['text'], persona)
            
            job_score = 0
            if job_description:
                job_score = self._calculate_job_relevance(chunk['text'], job_description)
            
            chunk['final_score'] = (
                chunk['rerank_score'] + 
                persona_weight * persona_score + 
                job_weight * job_score
            )
            
            chunk['persona_score'] = persona_score
            chunk['job_score'] = job_score
        
        ranked_chunks = sorted(candidate_chunks, key=lambda x: x['final_score'], reverse=True)
        return ranked_chunks[:top_k_rerank]

    def _generate_main_sections(self, query, persona, job_description=None, max_sections=7, 
                               word_count_weight=0.2, ideal_word_count=8):
        """
        Step 7: Main section extraction limited to top 7 most relevant sections.
        Extract sections from all labeled chunks (TITLE, H1, H2, H3) and return only the most relevant ones.
        Now includes preference for shorter texts.
        """
        print(f"Extracting top {max_sections} most relevant main sections from labeled chunks...")
        print(f"Word count preference: ideal={ideal_word_count} words, weight={word_count_weight}")
        
        # Get all chunks that are labeled as headers (not NONE)
        header_chunks = [chunk for chunk in self.labeled_chunks 
                        if chunk.get("predicted_label", "NONE") in ["TITLE", "H1", "H2", "H3"]]
        
        if not header_chunks:
            print("No header chunks found.")
            return []

        # Score headers against enhanced query
        enhanced_query = self._construct_enhanced_query(query, persona, job_description)
        header_texts = [chunk["text"] for chunk in header_chunks]
        header_embeddings = self.bi_encoder.encode(header_texts, convert_to_tensor=False)
        query_embedding = self.bi_encoder.encode([enhanced_query], convert_to_tensor=False)[0]

        # Calculate comprehensive relevance scores
        scored_sections = []
        seen = set()
        
        for i, chunk in enumerate(header_chunks):
            # Calculate semantic similarity
            similarity = float(np.dot(query_embedding, header_embeddings[i]) / 
                             (np.linalg.norm(query_embedding) * np.linalg.norm(header_embeddings[i])))
            
            # Calculate persona match
            persona_score = 0
            if persona:
                persona_score = self._calculate_persona_match(chunk["text"], persona)
            
            # Calculate job relevance
            job_score = 0
            if job_description:
                job_score = self._calculate_job_relevance(chunk["text"], job_description)
            
            # Calculate word count preference score
            word_count_score = self._calculate_word_count_penalty(chunk["text"], ideal_word_count)
            
            # Calculate combined relevance score with word count preference
            # Adjusted weights to accommodate word count preference
            base_weight = 1.0 - word_count_weight
            semantic_weight = 0.2 * base_weight
            persona_weight = 0.3 * base_weight
            job_weight = 0.5 * base_weight
            
            combined_score = (
                semantic_weight * similarity + 
                persona_weight * persona_score + 
                job_weight * job_score +
                word_count_weight * word_count_score
            )
            
            # Create unique key for deduplication
            key = (chunk["page_number"], chunk["text"])
            if key in seen:
                continue
            
            scored_sections.append({
                "document": chunk["metadata"]["doc_name"],
                "section_title": chunk["text"],
                "relevance_score": combined_score,
                "semantic_similarity": similarity,
                "persona_score": persona_score,
                "job_score": job_score,
                "word_count_score": word_count_score,
                "word_count": len(chunk["text"].split()),
                "page_number": chunk["page_number"],
                "predicted_label": chunk.get("predicted_label", "NONE"),
                "chunk_index": self.labeled_chunks.index(chunk)  # Store index for later use
            })
            seen.add(key)

        # Sort by combined relevance score and take top N
        scored_sections = sorted(scored_sections, key=lambda x: x["relevance_score"], reverse=True)
        top_sections = scored_sections[:max_sections]
        
        # Log the selection results for debugging
        print(f"Top {len(top_sections)} sections selected:")
        for i, section in enumerate(top_sections, 1):
            print(f"  {i}. '{section['section_title'][:50]}...' "
                  f"(words: {section['word_count']}, "
                  f"word_score: {section['word_count_score']:.3f}, "
                  f"total_score: {section['relevance_score']:.3f})")
        
        # Assign importance ranks and clean up scores for final output
        extracted_sections = []
        for rank, section in enumerate(top_sections, 1):
            final_section = {
                "document": section["document"],
                "section_title": section["section_title"],
                "importance_rank": rank,
                "page_number": section["page_number"],
                "chunk_index": section["chunk_index"]  # Keep for subsection processing
            }
            extracted_sections.append(final_section)
        
        print(f"Extracted top {len(extracted_sections)} most relevant main sections out of {len(scored_sections)} total header sections.")
        return extracted_sections

    def _generate_dynamic_subsections(self, main_sections, ranked_chunks):
        """
        Step 8: Generate subsections separately for each main section, even if they're from the same PDF.
        """
        print("Generating subsections for each selected main section separately...")
        
        if not main_sections:
            return []

        subsection_analysis = []

        # Process each main section individually
        for section in main_sections:
            section_doc = section["document"]
            section_page = section["page_number"]
            section_title = section["section_title"]
            
            print(f"Processing subsections for: '{section_title}' (Page {section_page}, {section_doc})")
            
            # Filter ranked chunks for this specific section
            # Include chunks from the same document and nearby pages (±1 page)
            section_chunks = []
            for chunk in ranked_chunks:
                chunk_doc = chunk['metadata']['doc_name']
                chunk_page = chunk['metadata']['page_number']
                
                # Include chunk if it's from the same document and nearby pages
                if chunk_doc == section_doc and abs(chunk_page - section_page) <= 1:
                    section_chunks.append(chunk)

            if len(section_chunks) < 1:
                print(f"  Not enough chunks for subsection analysis (found {len(section_chunks)} chunks)")
                continue

            # Generate embeddings for this section's chunks
            try:
                result_embeddings = np.array([self.chunk_embeddings[chunk['original_index']] for chunk in section_chunks])
                clusters = util.community_detection(result_embeddings, min_community_size=2, threshold=0.45)

                section_subsections = []
                
                for i, cluster in enumerate(clusters):
                    if len(cluster) < 2:  # Reduced minimum cluster size
                        continue

                    cluster_chunks = [section_chunks[j] for j in cluster]
                    full_text = " ".join([chunk['text'] for chunk in cluster_chunks])

                    try:
                        if len(full_text.split()) > 40:
                            summary = self.summarizer(full_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                        else:
                            summary = full_text
                    except Exception as e:
                        print(f"[ERROR] Summarization failed for section '{section_title}': {e}")
                        summary = full_text[:300] + "..."

                    pages = sorted(list(set([chunk['metadata']['page_number'] for chunk in cluster_chunks])))

                    subsection_data = {
                        "document": section_doc,
                        "refined_text": summary,
                        "page_number": pages[0] if len(pages) == 1 else pages,
                    }
                    section_subsections.append(subsection_data)

                if section_subsections:
                    subsection_analysis.extend(section_subsections)
                    print(f"  Generated {len(section_subsections)} subsections for '{section_title}'")
                else:
                    print(f"  No valid subsections generated for '{section_title}'")
                    
            except Exception as e:
                print(f"[ERROR] Failed to process subsections for section '{section_title}': {e}")
                continue

        print(f"Generated {len(subsection_analysis)} total subsections across all main sections.")
        return subsection_analysis

    def process_query(self, query, persona, input_documents, job_description=None, 
                     persona_weight=0.3, job_weight=0.2, max_sections=7,
                     word_count_weight=0.2, ideal_word_count=8):
        print("\n--- Processing Query ---")

        print("Step 1/4: Enhanced search and reranking...")
        ranked_chunks = self._search_and_rerank(
            query=query, 
            persona=persona, 
            job_description=job_description,
            persona_weight=persona_weight,
            job_weight=job_weight
        )

        print("Step 2/4: Main section extraction...")
        extracted_sections = self._generate_main_sections(
            query, persona, job_description, max_sections=max_sections,
            word_count_weight=word_count_weight, ideal_word_count=ideal_word_count
        )

        print("Step 3/4: Subsection analysis for each selected section...")
        subsection_analysis = self._generate_dynamic_subsections(extracted_sections, ranked_chunks)

        print("Step 4/4: Assembling final output...")
        # Remove internal fields from final output
        for section in extracted_sections:
            if 'chunk_index' in section:
                del section['chunk_index']

        final_output = {
            "metadata": {
                "input_documents": input_documents,
                "persona": persona,
                "job_to_be_done": query,
                "max_sections_extracted": max_sections,
                "total_sections_found": len(extracted_sections),
                "total_subsections_generated": len(subsection_analysis),
                "word_count_preference": {
                    "ideal_word_count": ideal_word_count,
                    "word_count_weight": word_count_weight
                },
                "processing_timestamp": datetime.now(timezone.utc).isoformat()
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }

        print("--- Query Processed Successfully ---")
        return final_output

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process PDF documents with custom query and persona',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Plan a trip of 4 days for a group of 10 college friends" "Travel Planner"
  python main.py "Create fillable forms for onboarding" "HR professional"
  python main.py "Prepare a vegetarian buffet menu" "Food Contractor"
        """
    )
    
    parser.add_argument('query', 
                       help='The query/job to be done (enclosed in quotes if contains spaces)')
    parser.add_argument('persona', 
                       help='The persona/role (enclosed in quotes if contains spaces)')
    parser.add_argument('--max-sections', 
                       type=int, 
                       default=7, 
                       help='Maximum number of sections to extract (default: 7)')
    parser.add_argument('--word-count-weight', 
                       type=float, 
                       default=0.2, 
                       help='Weight for word count preference (default: 0.2)')
    parser.add_argument('--ideal-word-count', 
                       type=int, 
                       default=8, 
                       help='Ideal number of words for section titles (default: 8)')
    parser.add_argument('--persona-weight', 
                       type=float, 
                       default=0.3, 
                       help='Weight for persona matching (default: 0.3)')
    parser.add_argument('--job-weight', 
                       type=float, 
                       default=0.7, 
                       help='Weight for job relevance (default: 0.7)')
    
    return parser.parse_args()

def main():
    """
    Main function to process PDFs with command line arguments.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"Query: {args.query}")
    print(f"Persona: {args.persona}")
    print(f"Max sections: {args.max_sections}")
    print(f"Word count weight: {args.word_count_weight}")
    print(f"Ideal word count: {args.ideal_word_count}")
    print(f"Persona weight: {args.persona_weight}")
    print(f"Job weight: {args.job_weight}")
    
    processor = DocumentProcessor(MODEL_PATHS)
    
    # Step 1: Get input PDFs and build index (Steps 2-5 are handled inside)
    processor.build_index_from_directory(PDF_DATA_PATH)
    
    INPUT_DOCS = [f for f in os.listdir(PDF_DATA_PATH) if f.endswith('.pdf')]
    
    # Process query (Steps 6-8 are handled inside)
    output_json = processor.process_query(
        query=args.query, 
        persona=args.persona, 
        input_documents=INPUT_DOCS,
        persona_weight=args.persona_weight,
        job_weight=args.job_weight,
        max_sections=args.max_sections,
        word_count_weight=args.word_count_weight,
        ideal_word_count=args.ideal_word_count
    )
    
    # Step 9: Save output in the same format
    output_filename = 'output.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    
    print(f"\nOutput saved to {output_filename} with top {args.max_sections} most relevant sections")
    print(f"Word count preference: ideal={args.ideal_word_count} words, weight={args.word_count_weight}")

if __name__ == '__main__':
    main()