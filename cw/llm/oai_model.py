from typing import List, Optional, Dict, Any, Union
import time
import asyncio
import os
import json
from dotenv import load_dotenv

import openai
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from llm_model import LlmModel, Message, LlmResponse, ToolCall
from console_logger import console_logger


class OaiModel(LlmModel):
    """OpenAI implementation of the LlmModel interface using the official OpenAI Python SDK."""
    
    OAI_MODEL = "o4-mini-2025-04-16"
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None, 
                 temperature: float = 0.7, max_tokens: Optional[int] = None,
                 base_url: Optional[str] = None):
        """
        Initialize the OpenAI model.
        
        Args:
            model: The OpenAI model to use (e.g., "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo")
            api_key: OpenAI API key (will use OPENAI_API_KEY env var if not provided)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            base_url: Optional base URL for API calls
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens or 4096
        
        # Initialize synchronous client
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        elif os.getenv("OPENAI_API_KEY"):
            client_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
        
        if base_url:
            client_kwargs["base_url"] = base_url
            
        self.client = OpenAI(**client_kwargs)
        
        # Initialize async client
        self.async_client = AsyncOpenAI(**client_kwargs)

    def _convert_messages_to_openai(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert our Message format to OpenAI's format."""
        openai_messages = []
        
        for msg in messages:
            openai_msg = {"role": msg.role}
            
            # Handle content
            if msg.content:
                openai_msg["content"] = msg.content
            
            # Handle tool calls in assistant messages
            if msg.tool_calls:
                openai_msg["tool_calls"] = []
                for tool_call in msg.tool_calls:
                    openai_msg["tool_calls"].append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.get("name", ""),
                            "arguments": tool_call.function.get("arguments", "{}")
                        }
                    })
            
            # Handle tool call responses
            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id
            
            openai_messages.append(openai_msg)
        
        return openai_messages

    def _convert_tools_to_openai(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Convert tool definitions to OpenAI format."""
        if not tools:
            return None
            
        openai_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                openai_tools.append(tool)  # OpenAI format is already compatible
        
        return openai_tools

    def _convert_openai_response(self, response: ChatCompletion, latency: float) -> LlmResponse:
        """Convert OpenAI response to our LlmResponse format."""
        choice = response.choices[0]
        message = choice.message
        
        content = message.content
        tool_calls = []
        
        # Handle tool calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tool_call.id,
                    type=tool_call.type,
                    function={
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                ))
        
        # Extract usage information
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        
        return LlmResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            latency_seconds=latency
        )

    def complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                tool_choice: Optional[Union[str, Dict[str, Any]]] = None, **kwargs) -> LlmResponse:
        """Synchronous completion using OpenAI models."""
        start_time = time.time()
        
        try:
            openai_messages = self._convert_messages_to_openai(messages)
            openai_tools = self._convert_tools_to_openai(tools)
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **kwargs
            }
            
            if openai_tools:
                request_params["tools"] = openai_tools
                
            if tool_choice:
                request_params["tool_choice"] = tool_choice
            
            # Make the API call
            response = self.client.chat.completions.create(**request_params)
            
            latency = time.time() - start_time
            llm_response = self._convert_openai_response(response, latency)
            
            # Debug logging
            self._debug_print_llm_response(llm_response)
            
            return llm_response
            
        except Exception as e:
            console_logger.log(f"Error in OpenAI completion: {str(e)}", "error")
            raise

    async def async_complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                           tool_choice: Optional[Union[str, Dict[str, Any]]] = None, **kwargs) -> LlmResponse:
        """Async completion using OpenAI models."""
        start_time = time.time()
        
        try:
            openai_messages = self._convert_messages_to_openai(messages)
            openai_tools = self._convert_tools_to_openai(tools)
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **kwargs
            }
            
            if openai_tools:
                request_params["tools"] = openai_tools
                
            if tool_choice:
                request_params["tool_choice"] = tool_choice
            
            # Make the async API call
            response = await self.async_client.chat.completions.create(**request_params)
            
            latency = time.time() - start_time
            llm_response = self._convert_openai_response(response, latency)
            
            # Debug logging
            self._debug_print_llm_response(llm_response)
            
            return llm_response
            
        except Exception as e:
            console_logger.log(f"Error in OpenAI async completion: {str(e)}", "error")
            raise

    def stream_complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                       tool_choice: Optional[Union[str, Dict[str, Any]]] = None, **kwargs):
        """Streaming completion using OpenAI models."""
        try:
            openai_messages = self._convert_messages_to_openai(messages)
            openai_tools = self._convert_tools_to_openai(tools)
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True,
                **kwargs
            }
            
            if openai_tools:
                request_params["tools"] = openai_tools
                
            if tool_choice:
                request_params["tool_choice"] = tool_choice
            
            # Return the streaming response
            return self.client.chat.completions.create(**request_params)
            
        except Exception as e:
            console_logger.log(f"Error in OpenAI streaming: {str(e)}", "error")
            raise


# Example usage and initialization
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Initialize the model
    oai_model = OaiModel(
        model="gpt-4o",
        temperature=0.7
    )
    
    # Test basic completion
    messages = [
        Message(role="user", content="What is the capital of France?")
    ]
    
    try:
        response = oai_model.complete(messages)
        print(f"Response: {response.content}")
        print(f"Usage: {response.usage}")
        print(f"Latency: {response.latency_seconds}s")
        
        # Test with tools
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
            }
        ]
        
        tool_messages = [
            Message(role="user", content="What's the weather like in New York?")
        ]
        
        tool_response = oai_model.complete(tool_messages, tools=tools, tool_choice="auto")
        print(f"\nTool Response: {tool_response.content}")
        if tool_response.tool_calls:
            print(f"Tool calls made: {len(tool_response.tool_calls)}")
            for i, tool_call in enumerate(tool_response.tool_calls):
                print(f"  Tool {i+1}: {tool_call.function}")
        
        # Test async completion
        async def test_async():
            async_response = await oai_model.async_complete(messages)
            print(f"\nAsync Response: {async_response.content}")
            
            # Test batch completion
            batch_requests = [
                [Message(role="user", content="What is 2+2?")],
                [Message(role="user", content="What is the largest planet?")],
                [Message(role="user", content="Who wrote Romeo and Juliet?")]
            ]
            
            batch_responses = await oai_model.batch_complete(batch_requests)
            for i, resp in enumerate(batch_responses):
                print(f"Batch response {i+1}: {resp.content}")
        
        # Run async test
        asyncio.run(test_async())
        
        # Test streaming
        print("\n" + "="*50)
        print("Testing streaming:")
        
        stream_messages = [
            Message(role="user", content="Tell me a short story about a robot.")
        ]
        
        stream = oai_model.stream_complete(stream_messages)
        print("Streaming response: ", end="", flush=True)
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()  # New line after streaming
        
    except Exception as e:
        print(f"Error: {e}")