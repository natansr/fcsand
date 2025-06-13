# A Flexible and Configurable System for Author Name Disambiguation

Author Name Disambiguation (AND) is essential for ensuring the integrity of bibliographic databases, particularly under data sparsity and ambiguity. The system implements a configurable AND system that integrates transformer-based embeddings (MiniLM), Graph Convolutional Networks (GCN), and hierarchical clustering. It allows users to adjust GCN depth, training epochs, and embedding models to handle datasets with diverse structural and semantic characteristics.

---

## Features

- Configurable transformer-based embeddings (MiniLM, Specter, SciBERT, and others)
- Adjustable GCN depth and training epochs
- Modular processing pipeline with GUI interface
- Supports datasets with varying metadata availability
- Evaluation using multiple AND metrics

---

## NLP Models

fcsand allows the user to select different pre-trained language models to generate embeddings. Available models include:

- `sentence-transformers/all-MiniLM-L6-v2` (default and recommended)
- `allenai/specter`
- `allenai/scibert_scivocab_uncased`



The model can be selected directly via the GUI before running the embedding extraction step.

---

## GUI

The system includes a user-friendly GUI that enables running the complete pipeline without coding. Users can load datasets, preprocess files, extract embeddings, build graphs, train GCN models, and apply clustering.

<p align="center">
  <img src="image.png" alt="GUI" width="600"/>
</p>



---

## Modules

- **Pre-processing**: Filters and structures JSON files based on available metadata.
- **Embeddings**: Extracts transformer-based embeddings from user-selected NLP models.
- **Graph Construction**: Builds heterogeneous graphs including authors, titles, abstracts, keywords, venues, and affiliations.
- **GCN**: Learns relational representations through Graph Convolutional Networks.
- **Clustering**: Applies hierarchical clustering on learned embeddings.
- **Evaluation**: Computes AND metrics: pP, pR, pF1, ACP, AAP, K-Metric, and B-cubed.

---

## Installation

Requirements:

- Python 3.10+
- PyTorch
- PyTorch Geometric
- HuggingFace Transformers
- scikit-learn
- ttkbootstrap
- tqdm
- pandas
- networkx

Clone and install:

```bash
git clone https://github.com/your_repository/fcsand.git
cd fcsand
pip install -r requirements.txt
```

---

## Running the System

Start the GUI:

```bash
python gui.py
```

The GUI allows you to execute:

- Metadata pre-processing
- Embedding extraction (with NLP model selection)
- Heterogeneous graph construction
- GCN training and tuning
- Clustering and evaluation

---

## Datasets

The system supports the following datasets (used in our experiments):

- AMiner-12
- DBLP
- LAGOS-AND

Example datasets are available for download:

[Download Datasets (Google Drive)](https://drive.google.com/drive/folders/1jxtOWCOlS6vX6ewIQYNHmaMEZncQrkyW?usp=drive_link)

Datasets should be placed inside the `datasets/` directory, containing JSON files with the following structure:

```json
{
  "id": "doc1",
  "title": "...",
  "abstract": "...",
  "venue": "...",
  "coauthors": ["..."],
  "keywords": ["..."],
  "label": "real_author_id"
}
```

---

## Notes

- The system includes full evaluation with Pairwise, Cluster-based, and B-cubed metrics.
- All modules can be executed independently via the GUI.
- Designed for reproducibility, scalability, and experimentation.

---

## Citation

...
