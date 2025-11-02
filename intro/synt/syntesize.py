# generate_goldens_example.py
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset

# Initialize synthesizer
synthesizer = Synthesizer()

# Generate goldens from multiple document types
# Start with just the required parameter to see what works
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=[
        'example.txt'
        # 'example.docx',
        # 'example.pdf'
    ]
)

# Create a dataset and save to JSON
dataset = EvaluationDataset(goldens=goldens)
dataset.save_as(
    file_type='json',
    directory='./',
    file_name='generated_goldens'
)

print(f"Generated {len(goldens)} golden test cases")
print(f"Saved to: generated_goldens.json")
