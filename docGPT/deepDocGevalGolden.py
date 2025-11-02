"""
DeepEval RAG Evaluation Example

This script demonstrates how to evaluate a Retrieval-Augmented Generation (RAG)
system using DeepEval's GEval metric. The RAG system retrieves relevant chunks
from a knowledge base and generates responses based on the retrieved context.

What it demonstrates:
- Load and chunk documents for retrieval
- Create embeddings for semantic search
- Implement a simple RAG function (doc_gpt)
- Define golden test cases with expected outputs
- Evaluate RAG outputs using GEval with explicit evaluation steps
- Compare actual vs expected outputs for correctness

Requirements:
- deepeval
- sentence-transformers
- scikit-learn
- numpy
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate

# ----------------------
# Load document
# ----------------------
def load_document(file_path="docs/knowledge.txt"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return text

document = load_document()
print("Loaded document")

# ----------------------
# Split document into chunks for retrieval
# ----------------------
# This function breaks the document into smaller pieces that can be
# individually retrieved and compared against queries.
def split_into_chunks(text, chunk_size=100):
    sentences = text.split(".")
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += sentence + "."
        else:
            chunks.append(current.strip())
            current = sentence + "."
    if current:
        chunks.append(current.strip())
    return chunks

# ----------------------
# Create embeddings for semantic search
# ----------------------
# Convert text chunks into numerical vectors that capture semantic meaning.
# These embeddings allow us to find similar content using cosine similarity.
chunks = split_into_chunks(document)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embedding_model.encode(chunks)

# ----------------------
# Retrieve most relevant chunk based on query
# ----------------------
# This is the "retrieval" part of RAG - find the most relevant document chunk
# for a given query using cosine similarity between embeddings.
def retrieve_context(query):
    query_emb = embedding_model.encode([query])
    scores = cosine_similarity(query_emb, chunk_embeddings)[0]
    best_chunk = chunks[np.argmax(scores)]
    return best_chunk

# ----------------------
# RAG function: retrieves context and generates response
# ----------------------
# This simulates a RAG system that:
# 1. Takes a user query
# 2. Retrieves relevant context from the knowledge base
# 3. Generates a response based on that context
def doc_gpt(query):
    context = retrieve_context(query)
    return f"Based on the document: {context}"

# ----------------------
# 1) Define goldens (reference answers)
# ----------------------
# These are your "gold standard" expected outputs for each input query.
# In a real evaluation, these would come from human-annotated data or
# verified correct answers.
goldens = [
    {
        "id": "q1",
        "input": "Where is the Eiffel Tower located?",
        "expected_output": "The Eiffel Tower is located in Paris, France."
    },
    {
        "id": "q2",
        "input": "What is the highest mountain in the world?",
        "expected_output": "Mount Everest is the highest mountain in the world."
    },
    {
        "id": "q3",
        "input": "Who is the current president of the United States?",
        "expected_output": "Joe Biden is current president of the United States."
    }
]

# ----------------------
# 2) Collect actual outputs from doc_gpt
# ----------------------
# Run each query through our RAG system to get the actual outputs.
# These will be compared against the expected outputs during evaluation.
actual_outputs = {}
for g in goldens:
    actual_outputs[g["id"]] = doc_gpt(g["input"])
    print(f"Query: {g['input']}")
    print(f"Output: {actual_outputs[g['id']]}")
    print("-" * 60)

# ----------------------
# 3) Build test cases
# ----------------------
# Create LLMTestCase objects (DeepEval's test format) that combine:
# - input query
# - actual output from our RAG system
# - expected output from goldens
# - metadata for tracking
test_cases = []
for g in goldens:
    tc = LLMTestCase(
        input=g["input"],
        actual_output=actual_outputs[g["id"]],
        expected_output=g["expected_output"],
        metadata={"golden_id": g["id"]}
    )
    test_cases.append(tc)

# ----------------------
# 4) Create GEval metric
# ----------------------
# GEval uses an LLM as a judge to evaluate outputs based on explicit criteria.
# The evaluation_steps define exactly what the judge should check.
# This is more flexible than simple string matching and can handle semantic similarity.
metric = GEval(
    name="Correctness",
    model="gpt-4o-mini",  # LLM used as judge (requires OPENAI_API_KEY)
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    evaluation_steps=[
        "Compare the factual content of 'actual_output' with 'expected_output'.",
        "Check if the answer includes correct factual information from the context.",
        "Ignore differences in phrasing or grammar."
    ],
    threshold=0.7  # Minimum score (0-1) to pass
)

# ----------------------
# 5) Run evaluation
# ----------------------
# The 'evaluate' call runs the metric across all test cases and returns
# structured results with scores, pass/fail status, and reasoning.
print("\n" + "=" * 60)
print("Running DeepEval evaluation...")
print("=" * 60)
results = evaluate(test_cases=test_cases, metrics=[metric])
print("Evaluation complete!")

# ----------------------
# Next steps
# ----------------------
# - Add more test cases to cover edge cases
# - Experiment with different chunking strategies
# - Try different embedding models for retrieval
# - Add more GEval metrics (e.g., Relevance, Completeness)
# - Integrate with your production RAG pipeline
