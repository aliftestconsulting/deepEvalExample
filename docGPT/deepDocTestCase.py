import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate

# Load document
def load_document(file_path="docs/knowledge.txt"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return text

document = load_document()
print("Loaded document")

# Split document into chunks for retrieval
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

# Create embeddings for semantic search
chunks = split_into_chunks(document)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embedding_model.encode(chunks)

# Retrieve most relevant chunk based on query
def retrieve_context(query):
    query_emb = embedding_model.encode([query])
    scores = cosine_similarity(query_emb, chunk_embeddings)[0]
    best_chunk = chunks[np.argmax(scores)]
    return best_chunk

# RAG function: retrieves context and generates response
def doc_gpt(query):
    context = retrieve_context(query)
    return f"Based on the document: {context}"

# Define evaluation metric
metric = GEval(
    name="Relevance",
    model="gpt-4o-mini",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    criteria="Evaluate whether the model output correctly answers the question based on the document.",
    evaluation_steps=[
        "Check if the answer includes correct factual information from the context.",
        "Ignore differences in phrasing or grammar."
    ],
    threshold=0.7  # Minimum score (0-1) to pass
)

# Define test cases
test_cases = [
    LLMTestCase(
        input="Where is the Eiffel Tower located?",
        actual_output=doc_gpt("Where is the Eiffel Tower located?"),
        expected_output="The Eiffel Tower is located in Paris, France."
    ),
    LLMTestCase(
        input="What is the highest mountain in the world?",
        actual_output=doc_gpt("What is the highest mountain in the world?"),
        expected_output="Mount Everest is the highest mountain in the world."
    )
]

# Run evaluation
print("Running DeepEval...")
results = evaluate(test_cases, [metric])
print("Evaluation complete!")
