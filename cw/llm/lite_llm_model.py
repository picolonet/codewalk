from typing import List, Optional, Dict, Any, Union
import litellm
from litellm import completion, acompletion
from litellm.types.utils import ModelResponse
from litellm.types.utils import StreamingChoices
import os
import time
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from console_logger import console_logger
from util.data_logger import get_data_logger
from llm.llm_model import LlmModel, Message, LlmResponse
from llm.llm_common import ToolCall, ToolCallResponse


class LiteLlmModel(LlmModel):
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None,
     temperature: float = 0.7, max_tokens: Optional[int] = None):
        super().__init__("lite_llm_claude")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        #litellm._turn_on_debug()
        
        # Setup LiteLLM error logging
        self._setup_litellm_logging()
        
        if api_key:
            litellm.api_key = api_key
        if base_url:
            litellm.api_base = base_url

        self.custom_headers = {
            "anthropic-beta": "prompt-caching-2024-07-31",
        }
            
    def _setup_litellm_logging(self):
        """Setup LiteLLM logging to capture errors to a file in logs/ directory"""
        self._setup_llm_logging()
        # Set LiteLLM to use our error logger
        litellm.logger = self.error_logger


    def complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None, trace_name: Optional[str] = None, **kwargs) -> LlmResponse:
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
        # self._debug_print_formatted_messages(formatted_messages)
        
        # Create Langfuse trace
        trace = self._create_trace(name=trace_name or "llm_completion")
        
        completion_kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "stream": False,  # Always non-streaming in complete()
            "headers": self.custom_headers,
            **kwargs
        }
        
        if self.max_tokens:
            completion_kwargs["max_tokens"] = self.max_tokens
        
        if tools:
            completion_kwargs["tools"] = tools
        
        if tool_choice:
            completion_kwargs["tool_choice"] = tool_choice
        

        generation = self._trace_generation(trace=trace, name="litellm_completion", model=self.model, formatted_messages=formatted_messages,
            temperature=self.temperature, max_tokens=self.max_tokens, tools=tools, tool_choice=tool_choice)
        
        # Measure latency
        start_time = time.time()
        try:
            response = completion(**completion_kwargs)
        except Exception as e:
            # Log the error with context
            self.error_logger.error(f"LiteLLM completion error: {str(e)}", extra={
                'model': self.model,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'tools_count': len(tools) if tools else 0,
                'messages_count': len(formatted_messages)
            })
            raise  # Re-raise the exception to maintain existing behavior
        end_time = time.time()
        latency = end_time - start_time
        
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
        final_response = LlmResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            latency_seconds=latency
        )
        

        self._trace_end(generation=generation, output=message.content, prompt_tokens=final_response.get_prompt_tokens(),
            completion_tokens=final_response.get_completion_tokens(), total_tokens=final_response.get_total_tokens())
        
        self._debug_print_llm_response(final_response)
        data_logger = get_data_logger()
        #print(self.model_name, type(self.model_name))

        data_logger.log_stats(model_name=self.model_name, prompt_tokens=final_response.get_prompt_tokens(),
                 completion_tokens=final_response.get_completion_tokens(), latency_seconds=final_response.get_latency_seconds(),
                  operation="tool_call" if tool_calls != None else "completion")
        return final_response

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
        
        # Measure initial connection latency for streaming
        start_time = time.time()
        stream = completion(**completion_kwargs)
        # Note: For streaming, this only measures the initial connection time
        # Full latency would need to be measured by the caller as chunks arrive
        return stream

    async def async_complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None, trace_name: Optional[str] = None, **kwargs) -> LlmResponse:
        """Async version of complete method for parallel processing."""
        
        formatted_messages = self._format_messages(messages)

        trace_name = trace_name or "llm_async_completion"
        
        # Create Langfuse trace
        trace = self._create_trace(name=trace_name)
        
        completion_kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "stream": False,
            **kwargs
        }
        
        if self.max_tokens:
            completion_kwargs["max_tokens"] = self.max_tokens
        
        if tools:
            completion_kwargs["tools"] = tools
        
        if tool_choice:
            completion_kwargs["tool_choice"] = tool_choice
        

        generation = self._trace_generation(trace=trace, name=trace_name, model=self.model, formatted_messages=formatted_messages,
            temperature=self.temperature, max_tokens=self.max_tokens, tools=tools, tool_choice=tool_choice)
        
        # Measure latency
        start_time = time.time()
        try:
            response = await acompletion(**completion_kwargs)
        except Exception as e:
            # Log the error with context
            self.error_logger.error(f"LiteLLM async completion error: {str(e)}", extra={
                'model': self.model,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'tools_count': len(tools) if tools else 0,
                'messages_count': len(formatted_messages)
            })
            raise  # Re-raise the exception to maintain existing behavior
        end_time = time.time()
        latency = end_time - start_time
        
        # Guard: ensure response is a ModelResponse (not a streaming object)
        if not isinstance(response, ModelResponse):
            raise RuntimeError(f"Expected ModelResponse from acompletion(), got {type(response)}.\n"
                               "This usually means a streaming object was returned.\n"
                               "Ensure 'stream' is False.")
        
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
        final_response = LlmResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            latency_seconds=latency
        )
        

        self._trace_end(trace=trace, generation=generation, output=message.content, prompt_tokens=final_response.get_prompt_tokens(),
            completion_tokens=final_response.get_completion_tokens(), total_tokens=final_response.get_total_tokens())
        
        self._debug_print_llm_response(final_response)
        return final_response


# Instantiate singleton global

# Get API key from .env file
load_dotenv()
api_key = os.environ.get("CODEWALKER_API_KEY")
base_url = os.environ.get("CODEWALKER_BASE_URL")
model_name = os.environ.get("CODEWALKER_MODEL_NAME") or "gpt-3.5-turbo"

lite_llm = LiteLlmModel(
    model=model_name,
    base_url=base_url,
    api_key=api_key,  # Or set OPENAI_API_KEY environment variable
    temperature=0.7
)


# Example usage of the above class
########################################################################################
if __name__ == "__main__":
    # Initialize the model
    llm = LiteLlmModel(
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
        print("Latency:", f"{response.latency_seconds:.3f}s" if response.latency_seconds else "Not measured")
        
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
            print(f"Latency: {final_response.latency_seconds:.3f}s" if final_response.latency_seconds else "Not measured")
        
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