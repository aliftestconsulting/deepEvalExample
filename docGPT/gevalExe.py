"""
DeepEval G-Eval example (Python)

This script shows a minimal, runnable example that uses DeepEval's GEval metric
with explicit goldens (expected outputs). It is written as a simple script you
can adapt to your project. Replace the placeholder model/keys with your own
LLM config if required.

Requirements
- deepeval (install from pip / your environment)

What it demonstrates
- Define goldens (reference expected outputs)
- Create LLMTestCase instances combining input, expected_output, and an
  actual_output (usually from your model)
- Instantiate a GEval metric with explicit evaluation_steps
- Run evaluate(...) to compute scores

NOTE: This example uses synchronous evaluation and dummy model outputs. In
production you'll collect real model outputs and likely set a different LLM
backend (e.g., an API key or a DeepEvalBaseLLM implementation).
"""

from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval import evaluate

# ----------------------
# 1) Define goldens (reference answers)
# ----------------------
# These are your "gold standard" expected outputs for each input.
goldens = [
    {
        "id": "g1",
        "input": "What if these shoes don't fit?",
        "expected_output": "You can return or exchange items within 30 days; see our returns page for details."
    },
    {
        "id": "g2",
        "input": "How do I reset my password?",
        "expected_output": "Open the app, go to Settings → Account → Reset Password; we'll send a reset link to your email."
    },
    {
        "id": "g3",
        "input": "Summarize the economic effects of inflation in one paragraph.",
        "expected_output": "Inflation erodes purchasing power, raises input costs for businesses, can slow consumer spending, and if persistent may cause central banks to hike interest rates—affecting investment and economic growth."
    }
]

# ----------------------
# 2) Collect actual outputs (from your model)
# ----------------------
# In real usage you'd generate these by calling your LLM / service. Here we
# use placeholder outputs to demonstrate evaluation.
actual_outputs = {
    "g1": "We accept returns within 30 days and offer free exchanges.",
    "g2": "Tap Settings > Reset Password and follow the emailed link.",
    "g3": "Inflation makes goods more expensive; central banks may raise rates which affects borrowing and spending."
}

# ----------------------
# 3) Build LLMTestCase objects (DeepEval's runtime test-case format)
# ----------------------
# Each LLMTestCase should include at minimum: input and actual_output. Since
# GEval is reference-based here, we also include expected_output from the
# golden.

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
# 4) Create a GEval metric
# ----------------------
# We provide evaluation_steps explicitly for deterministic behavior. These
# steps describe exactly what the LLM-as-a-judge should check.

correctness_metric = GEval(
    name="AnswerCorrectness",
    evaluation_steps=[
        "Compare the factual content of 'actual_output' with 'expected_output' and identify contradictions.",
        "Penalize missing critical details present in 'expected_output' (omit = -0.5).",
        "If the 'actual_output' conveys the same meaning and important details as 'expected_output', award full points."
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    # optional settings
    threshold=0.7,        # what counts as pass (0..1)
    model="gpt-4.1",     # default DeepEval model; change to match your environment
    verbose_mode=False
)

# You can define additional GEval metrics (e.g., Clarity, Tone) and run them
# together. For brevity we use a single metric here.

# ----------------------
# 5) Run evaluation
# ----------------------
# The 'evaluate' call will run the metric(s) across all test_cases and return
# structured results.

results = evaluate(test_cases=test_cases, metrics=[correctness_metric])

# ----------------------
# 6) Inspect results
# ----------------------
# 'results' is a tuple where the first element contains the test results.
# Each result has a test_case and metrics_data with scores and reasoning.

# Handle tuple return from evaluate()
test_results = results[0] if isinstance(results, tuple) else results

print("\n" + "=" * 60)
print("Detailed Results:")
print("=" * 60)

for idx, test_result in enumerate(test_results, 1):
    golden_id = test_result.metadata.get('golden_id', f'test_{idx}')
    print(f"\n[Test Case: {golden_id}]")
    print(f"Input: {test_result.input}")
    print(f"Expected: {test_result.expected_output}")
    print(f"Actual: {test_result.actual_output}")
    print(f"Success: {test_result.success}")
    
    # Display metric scores
    for metric_data in test_result.metrics_data:
        print(f"\nMetric: {metric_data.name}")
        print(f"Score: {metric_data.score:.3f}")
        if hasattr(metric_data, 'reason') and metric_data.reason:
            print(f"Reason: {metric_data.reason}")
    print("-" * 60)

# ----------------------
# Next steps / Integration
# ----------------------
# - Replace `actual_outputs` with outputs from your LLM inference pipeline.
# - Tweak evaluation_steps or criteria to better match your task's nuance.
# - Add more GEval metrics (Clarity, Tone, Safety) for multi-dimensional
#   evaluation.
# - Consider running evaluations asynchronously or in batches for large
#   datasets.

