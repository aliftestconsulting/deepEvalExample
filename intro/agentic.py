from deepeval.tracing import observe, update_current_trace
from deepeval.metrics import TaskCompletionMetric, ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval import evaluate
import os
import random

# --- Load knowledge base ---
def load_knowledge(file_path="docs/knowledge.txt"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Knowledge file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

knowledge_base = load_knowledge()

# --- Define tools (external functionalities) ---  
def fetch_weather(city: str) -> str:
    """Fetch weather from knowledge base with possible noise."""
    lines = knowledge_base.split("\n")
    
    # 20% chance of returning the wrong city to simulate imperfection
    if random.random() < 0.2:
        # Pick a random line (wrong city)
        wrong_line = random.choice([l for l in lines if l.strip() and city.lower() not in l.lower()])
        return wrong_line.split("is")[-1].strip() + f" in {city}!"
    
    # 80% correct - search for the city in knowledge base
    for line in lines:
        if city.lower() in line.lower():
            weather = line.split("is")[-1].strip().rstrip(".")
            return f"{weather} in {city}!"
    
    # Fallback if city not found
    return f"No weather data available for {city}."

@observe()
def call_weather_tool(city: str) -> str:
    result = fetch_weather(city)
    # Register the tool call in the trace  
    update_current_trace(name="fetch_weather", input={"city": city}, output=result)
    return result

# --- Define the agent (or conversational system) ---  
@observe()
def weather_agent(user_input: str) -> str:
    # 30% chance of misunderstanding user input
    if random.random() < 0.3:
        output = "I'm not sure what you mean, maybe check the forecast yourself?"
        return output

    # Simple parsing example
    if "weather" in user_input.lower():
        city = user_input.split("in")[-1].strip().rstrip("?")
        weather_str = call_weather_tool(city)
        output = f"The weather today: {weather_str}"
        return output
    else:
        output = "I can only help with weather inquiries."
        return output

# --- Simulate a test case ---  
user_input = "What is the weather in Jakarta?"
expected_output = "The weather today: Sunny in Jakarta!"

# Modified call to track tools
actual_output = weather_agent(user_input)

# Create proper ToolCall object
tool_call = ToolCall(
    name="fetch_weather",
    input={"city": "Jakarta"},
    output="Sunny in Jakarta!"
)

test_case = LLMTestCase(
    input=user_input,
    expected_output=expected_output,
    actual_output=actual_output,
    tools_called=[tool_call]
)

# --- Select metrics and run evaluation ---  
metrics = [
    TaskCompletionMetric(threshold=0.5),
    ArgumentCorrectnessMetric(threshold=0.5)
]

print("Running evaluation with tracing...")
evaluate(
    [test_case],
    metrics
)
print("Evaluation complete!")
