# PDF Processing System

A sophisticated document processing system that extracts, analyzes, and ranks content from PDF documents based on user queries and personas using advanced NLP techniques.

## 📁 Folder Structure

```
project_root/
├── main.py                     # Main processing script
├── utils/
│   ├── local_model.py         # Local heading model utilities
│   ├── bi_encoder/            # Sentence transformer bi-encoder model
│   ├── cross_encoder/         # Cross-encoder for reranking
│   ├── summarizer/            # Text summarization model
│   └── labeler/               # Document section labeling model
├── input/                     # 📄 Place your PDF files here
│   └── *.pdf                  # PDF documents to be processed
├── output/                    # Generated results
│   └── output.json           # Final processed results
├── chunk_data/               # Intermediate processing data
│   └── all_labeled_chunks.json # Labeled document chunks
└── README.md                 # This file
```

## 🚀 Main Functions & Libraries

### Core Functionality

The system processes PDF documents through a multi-stage pipeline:

1. **Document Chunking**: Extracts text blocks with metadata (position, formatting, etc.)
2. **Content Labeling**: Classifies chunks as TITLE, H1, H2, H3, or NONE
3. **Vector Indexing**: Creates semantic embeddings using FAISS for fast similarity search
4. **Query Processing**: Matches user queries with document content using persona-aware ranking
5. **Content Extraction**: Identifies most relevant sections and generates subsections

## 📄 PDF Input Requirements

### Input Directory Setup

- **Location**: All PDF files must be placed in the `./input/` directory
- **Format**: Only `.pdf` files are supported
- **Processing**: The system processes ALL PDF files found in the input directory

### ⚠️ Important: Clean Input Directory

Before running a new test case, you MUST delete all files from the input directory:

```bash
# Remove all files from input directory
rm -rf ./input/*

# Or manually delete all PDF files from the input folder
```

This ensures:

- No interference from previous test documents
- Clean processing environment
- Accurate results for your current test case
- Prevents memory issues with large document collections

## 🎭 Input Parameters

### Command Line Usage

```bash
python main.py "your query here" "your persona here"
```

### Interactive Mode

If no command line arguments are provided, the system will prompt for input:

```
Enter your query/job to be done: [Your query]
Enter your persona/role: [Your persona]
```

### Parameter Examples

#### Query (Job to be Done)

The specific task or information you're looking for:

- "Find information about machine learning algorithms"
- "Extract project management methodologies"
- "Identify security best practices"
- "Locate financial analysis techniques"

#### Persona (User Role)

Your role or perspective that influences content relevance:

- "Software Engineer"
- "Project Manager"
- "Security Analyst"
- "Financial Analyst"
- "Data Scientist"
- "Business Consultant"

### Advanced Configuration

The system includes several tunable parameters (set in main function):

| Parameter         | Default | Description                                  |
| ----------------- | ------- | -------------------------------------------- |
| max_sections      | 7       | Maximum number of main sections to extract   |
| word_count_weight | 0.2     | Weight for preferring shorter section titles |
| ideal_word_count  | 5       | Ideal word count for section titles          |
| persona_weight    | 0.3     | Influence of persona matching on relevance   |
| job_weight        | 0.7     | Influence of job/query matching on relevance |

## 🔧 Setup & Installation

### Prerequisites

```bash
pip install pymupdf faiss-cpu sentence-transformers transformers numpy
```

### Model Requirements

Ensure the following pre-trained models are available in the `./models/` directory:

- Bi-encoder model for semantic embeddings
- Cross-encoder model for reranking
- Summarization model for content condensation
- Labeler model for section classification

## 📊 Output Format

The system generates a JSON file (`./output/output.json`) containing:

```json
{
  "metadata": {
    "input_documents": ["document1.pdf", "document2.pdf"],
    "persona": "Software Engineer",
    "job_to_be_done": "Find machine learning algorithms",
    "max_sections_extracted": 7,
    "total_sections_found": 7,
    "total_subsections_generated": 15,
    "processing_timestamp": "2024-01-01T12:00:00Z"
  },
  "extracted_sections": [
    {
      "document": "document1.pdf",
      "section_title": "Machine Learning Overview",
      "importance_rank": 1,
      "page_number": 5
    }
  ],
  "subsection_analysis": [
    {
      "document": "document1.pdf",
      "refined_text": "Summary of subsection content...",
      "page_number": 5
    }
  ]
}
```

## 🏃‍♂️ Quick Start

1. **Prepare your environment:**

   ```bash
   mkdir -p input output chunk_data
   ```

2. **Clean input directory:**

   ```bash
   rm -rf ./input/*
   ```

3. **Add your PDF files:**

   ```bash
   cp your_documents/*.pdf ./input/
   ```

4. **Run the processor:**

   **Using Python directly:**

   ```bash
   python main.py "your query" "your persona"
   ```

   **Using Docker:**

   ```bash
   docker run --rm -v %cd%/input:/app/input -v %cd%/output:/app/output adobe-round1b \
   "Create fillable forms for onboarding" "HR professional"
   ```

5. **Check results:**
   ```bash
   cat ./output/output.json
   ```

## 💡 Tips for Best Results

- **Clear Queries**: Use specific, descriptive queries for better matching
- **Relevant Personas**: Choose personas that align with your intended use of the information
- **Clean Input**: Always start with a clean input directory for accurate results
- **Document Quality**: Higher quality PDFs with clear structure yield better results
- **Section Titles**: The system works best with documents that have clear headings and structure

## 🚨 Troubleshooting

- **No results**: Check if PDFs contain extractable text (not just images)
- **Memory issues**: Reduce the number of PDFs processed simultaneously
- **Model errors**: Ensure all required models are properly installed in `./models/`
- **Permission errors**: Verify read/write permissions for input, output, and chunk_data directories
