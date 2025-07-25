from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import litellm
from litellm import completion
from rich.console import Console
from rich.panel import Panel
import json
from litellm.types.utils import ModelResponse
from litellm.types.utils import StreamingChoices
import os
from dotenv import load_dotenv





class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Dict[str, Any]


class ToolCallResponse(BaseModel):
    id: str
    content: str
    tool_call_id: str


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class LlmResponse(BaseModel):
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: str
    usage: Dict[str, Any]


def format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
    """ Formats a list of messages for console output. """
    formatted_messages = []
    for msg in messages:
        formatted_msg: Dict[str, Any] = {"role": msg.role}

        if msg.content:
            formatted_msg["content"] = msg.content

        if msg.tool_calls:
            formatted_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": tc.function
                }
                for tc in msg.tool_calls
            ]

        if msg.tool_call_id:
            formatted_msg["tool_call_id"] = msg.tool_call_id

        formatted_messages.append(formatted_msg)

    return formatted_messages


class LlmModel:
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None,
     temperature: float = 0.7, max_tokens: Optional[int] = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if api_key:
            litellm.api_key = api_key
        if base_url:
            litellm.api_base = base_url

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        formatted_messages = []
        for msg in messages:
            formatted_msg: Dict[str, Any] = {"role": msg.role}
            
            if msg.content:
                formatted_msg["content"] = msg.content
            
            if msg.tool_calls:
                formatted_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": tc.function
                    }
                    for tc in msg.tool_calls
                ]
            
            if msg.tool_call_id:
                formatted_msg["tool_call_id"] = msg.tool_call_id
            
            formatted_messages.append(formatted_msg)
        
        return formatted_messages


    def _debug_print_formatted_messages(self, formatted_messages: List[Dict[str, Any]]) -> None:
        """Pretty prints formatted messages to terminal for debugging using rich."""

        console = Console()
        
        for i, msg in enumerate(formatted_messages):
            title = f"Message {i+1} ({msg.get('role', 'unknown role')})"
            json_str = json.dumps(msg, indent=2)
            console.print(Panel(json_str, title=title, border_style="blue"))



    def complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None, **kwargs) -> LlmResponse:
        """Completes a chat conversation using the configured LLM.

        Args:
            messages: List of Message objects representing the conversation history
            tools: Optional list of tool definitions that the model can use
            tool_choice: Optional specification of which tool the model should use
            **kwargs: Additional keyword arguments passed to the completion API

        Returns:
            LlmResponse containing the model's response, including:
                - content: The text response
                - tool_calls: Any tool calls made by the model
                - finish_reason: Why the model stopped generating
                - usage: Token usage statistics
        """

        formatted_messages = self._format_messages(messages)
        self._debug_print_formatted_messages(formatted_messages)
        
        completion_kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "stream": False,  # Always non-streaming in complete()
            **kwargs
        }
        
        if self.max_tokens:
            completion_kwargs["max_tokens"] = self.max_tokens
        
        if tools:
            completion_kwargs["tools"] = tools
        
        if tool_choice:
            completion_kwargs["tool_choice"] = tool_choice
        
        response = completion(**completion_kwargs)
        # Guard: ensure response is a ModelResponse (not a streaming object)
        if not isinstance(response, ModelResponse):
            raise RuntimeError(f"Expected ModelResponse from completion(), got {type(response)}.\n"
                               "This usually means a streaming object was returned.\n"
                               "Ensure 'stream' is False and do not use stream_complete for this call.")
        
        choice = response.choices[0]
        if isinstance(choice, StreamingChoices):
            raise RuntimeError("Received a StreamingChoices object in non-streaming mode. This should not happen.")
        message = choice.message
        
        tool_calls = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                func = tc.function
                if hasattr(func, 'model_dump'):
                    func = func.model_dump()
                elif hasattr(func, 'dict'):
                    func = func.dict()
                if not isinstance(func, dict):
                    func = {}
                tool_calls.append(ToolCall(id=tc.id, type=tc.type, function=func))
        
        usage = getattr(response, 'usage', {}) if hasattr(response, 'usage') else {}
        if not isinstance(usage, dict) and hasattr(usage, 'model_dump'):
            usage = usage.model_dump()
        finish_reason = choice.finish_reason if choice.finish_reason is not None else 'stop'
        return LlmResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage
        )

    def stream_complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ):
        formatted_messages = self._format_messages(messages)
        
        completion_kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "stream": True,
            **kwargs
        }
        
        if self.max_tokens:
            completion_kwargs["max_tokens"] = self.max_tokens
        
        if tools:
            completion_kwargs["tools"] = tools
        
        if tool_choice:
            completion_kwargs["tool_choice"] = tool_choice
        
        return completion(**completion_kwargs)


# Example usage of the above class
########################################################################################
if __name__ == "__main__":
    # Initialize the model
    llm = LlmModel(
        model="gpt-3.5-turbo",
        api_key="your-api-key-here",  # Or set OPENAI_API_KEY environment variable
        temperature=0.7
    )
    
    # Define some example tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "search_files",
                "description": "Search for files in a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Directory path to search in"
                        },
                        "pattern": {
                            "type": "string", 
                            "description": "File pattern to search for (e.g., '*.py')"
                        }
                    },
                    "required": ["directory", "pattern"]
                }
            }
        }
    ]
    
    # Create messages
    messages = [
        Message(
            role="system",
            content="You are a helpful assistant that can call tools to help users."
        ),
        Message(
            role="user", 
            content="What's the weather in San Francisco and can you search for Python files in the /home/user directory?"
        )
    ]
    
    # Make completion request with tools
    try:
        response = llm.complete(
            messages=messages,
            tools=tools,
            tool_choice="auto"  # Let the model decide which tools to use
        )
        
        print("Response content:", response.content)
        print("Finish reason:", response.finish_reason)
        print("Usage:", response.usage)
        
        # Check if model made tool calls
        if response.tool_calls:
            print(f"\nModel made {len(response.tool_calls)} tool call(s):")
            
            # Add assistant message with tool calls to conversation
            messages.append(Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls
            ))
            
            # Process each tool call and create responses
            tool_responses = []
            for i, tool_call in enumerate(response.tool_calls):
                print(f"  Tool call {i+1}:")
                print(f"    ID: {tool_call.id}")
                print(f"    Type: {tool_call.type}")
                print(f"    Function: {tool_call.function}")
                
                # Simulate tool execution based on function name
                function_name = tool_call.function.get("name", "")
                if function_name == "get_weather":
                    tool_result = "The weather in San Francisco is 72°F and sunny."
                elif function_name == "search_files":
                    tool_result = "Found 15 Python files: app.py, utils.py, models.py, test_main.py, config.py..."
                else:
                    tool_result = f"Tool {function_name} executed successfully."
                
                # Create ToolCallResponse object
                tool_response = ToolCallResponse(
                    id=f"resp_{tool_call.id}",
                    content=tool_result,
                    tool_call_id=tool_call.id
                )
                tool_responses.append(tool_response)
                
                # Add tool response message to conversation
                messages.append(Message(
                    role="tool",
                    content=tool_response.content,
                    tool_call_id=tool_response.tool_call_id
                ))
                
                print(f"    Response: {tool_response.content}")
            
            # Get final response after tool calls
            final_response = llm.complete(messages=messages)
            print(f"\nFinal response after tool execution:")
            print(f"Content: {final_response.content}")
            print(f"Finish reason: {final_response.finish_reason}")
        
    except Exception as e:
        print(f"Error: {e}")
        
    # Example of creating tool calls manually
    print("\n" + "="*50)
    print("Manual tool call creation example:")
    
    # Create a ToolCall manually
    manual_tool_call = ToolCall(
        id="call_123",
        type="function",
        function={
            "name": "calculate_sum",
            "arguments": '{"numbers": [1, 2, 3, 4, 5]}'
        }
    )
    
    # Create corresponding response
    manual_tool_response = ToolCallResponse(
        id="resp_123",
        content="The sum is 15",
        tool_call_id="call_123"
    )
    
    print(f"Manual tool call: {manual_tool_call}")
    print(f"Manual tool response: {manual_tool_response}")
    
    # Example of streaming
    print("\n" + "="*50)
    print("Streaming example:")
    
    stream_messages = [
        Message(role="user", content="Tell me a short story about a robot.")
    ]
    
    try:
        stream = llm.stream_complete(messages=stream_messages)
        print("Streaming response: ", end="", flush=True)
        for chunk in stream:
            if hasattr(chunk, 'choices') and chunk.choices:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        print(content, end="", flush=True)
        print()  # New line after streaming
        
    except Exception as e:
        print(f"Streaming error: {e}")
else:
    print("LlmModel class is not being run as the main module.")



# Get API key from .env file
load_dotenv()
api_key = os.environ.get("CODEWALKER_API_KEY")
base_url = os.environ.get("CODEWALKER_BASE_URL")
model_name = os.environ.get("CODEWALKER_MODEL_NAME") or "gpt-3.5-turbo"

llm = LlmModel(
    model=model_name,
    base_url=base_url,
    api_key=api_key,  # Or set OPENAI_API_KEY environment variable
    temperature=0.7
)




