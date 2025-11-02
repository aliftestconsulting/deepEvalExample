# Please pip install mcp before executing this code.
# Brief:
# User → [MCP Host (your app)]
#       ↳ [MCP Client → Weather API Server]
#       ↳ [MCP Client → Calendar Server]
#       ↳ [MCP Client → Knowledge Base]
#       ↓
#   LLM combines responses → Output
# Example:

# User : “What’s the weather in Tokyo tomorrow?”
# LLM internally decides: “I need to call the weather tool.”
# MCP Client (your app) Tool Call:
# weather_api.get_forecast(city="Tokyo", date="2025-11-02")


# Tool returns: "Sunny, 22°C"
# LLM replies: “Tomorrow in Tokyo it’ll be sunny, around 22°C.”
# → The LLM didn’t guess — it used a tool (API, DB, file, plugin, etc.).
# → This call is tracked and evaluated by DeepEval MCP metrics.


from deepeval.test_case import LLMTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MCPUseMetric, MCPTaskCompletionMetric
from deepeval import evaluate
from mcp.types import CallToolResult

# --- Dummy MCP tool simulation ---
def math_server_add(a, b):
    """Simulated MCP server performing addition."""
    return a + b

def mcp_host_addition_request(a, b):
    result = math_server_add(a, b)
    # Create CallToolResult with proper structure
    call_result = CallToolResult(
        content=[{"type": "text", "text": str(result)}],
        isError=False
    )
    tool_call = MCPToolCall(
        name="math_server_add",
        args={"a": a, "b": b},
        result=call_result,
    )
    return f"The answer is {result}", tool_call


# --- Define test input/output ---
user_input = "Add 3 and 5"
expected_output = "The answer is 8"
actual_output, tool_call = mcp_host_addition_request(3, 5)
# Ensure actual_output is a string (not tuple)
actual_output = str(actual_output)

# Use server_name instead of server_id
math_server = MCPServer(server_name="math_server")

# Build test case
test_case = LLMTestCase(
    input=user_input,
    expected_output=expected_output,
    actual_output=actual_output,
    mcp_tools_called=[tool_call],
    mcp_servers=[math_server],
)

# Metrics
# confirm that a tool named math_server_add was called (good usage)
mcp_use_metric = MCPUseMetric(threshold=0.5)
# confirm that the final text "The answer is 8" matches expectation
mcp_task_completion_metric = MCPTaskCompletionMetric(threshold=0.5)

# Run evaluation
evaluate(
    [test_case],
    [mcp_use_metric, mcp_task_completion_metric],
)
