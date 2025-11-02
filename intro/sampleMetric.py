from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

# Context must be a list of strings, not a single string
context = [
    "The Eiffel Tower is in Paris, France.",
    "It was built between 1887 and 1889."
]

test_case = LLMTestCase(
    input="When was the Eiffel Tower built?",
    actual_output="The Eiffel Tower was built in 1888 and designed by Leonardo da Vinci.",
    context=context  # Now it's a list of strings
)

metric = HallucinationMetric(threshold=0.7)

result = evaluate([test_case], [metric])
print(result)
