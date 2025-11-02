from deepeval.test_case import LLMTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MCPUseMetric
from deepeval import evaluate
from mcp.types import CallToolResult


# Simulated MCP server and tool
def math_server_add(a, b):
    """Simulated external math tool (MCP Server)."""
    return a + b


# Simulated chat sequence
conversation = [
    {"role": "user", "content": "Hi, can you add 3 and 5?"},
    {"role": "assistant", "content": "Sure! Let me use my math server to calculate that..."},
    {"role": "assistant_tool_call", "content": "Calling math_server_add(a=3, b=5)"},
    {"role": "tool_response", "content": "Result: 8"},
    {"role": "assistant", "content": "The answer is 8! Would you like to add more numbers?"},
]

# Define expected behavior
expected_final_output = "The answer is 8"
expected_tool_name = "math_server_add"

# Simulate what actually happened
actual_output = conversation[-1]["content"]
call_result = CallToolResult(
    content=[{"type": "text", "text": "8"}],
    isError=False
)
tool_call = MCPToolCall(
    name=expected_tool_name,
    args={"a": 3, "b": 5},
    result=call_result,
)

# Define the MCP server (old schema for your DeepEval 3.6.9)
math_server = MCPServer(server_name="math_server")

# Create the test case
test_case = LLMTestCase(
    input="\n".join([turn["content"] for turn in conversation]),
    expected_output=expected_final_output,
    actual_output=actual_output,
    mcp_tools_called=[tool_call],
    mcp_servers=[math_server],
    context=["multi-turn conversation: assistant must call math_server_add when doing arithmetic."],
)

# Define metrics
mcp_use_metric = MCPUseMetric(threshold=0.5)

# Evaluate
evaluate(
    [test_case],
    [mcp_use_metric],
)
