from typing import List, Optional, Dict, Any, Union
import time
import asyncio
import os
from dotenv import load_dotenv

import anthropic
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message as AnthropicMessage, MessageParam

from llm.llm_model import LlmModel, Message, LlmResponse
from llm.llm_common import ToolCall
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from console_logger import console_logger


class AnthropicModel(LlmModel):
    """Anthropic Claude implementation of the LlmModel interface using the official Anthropic Python SDK."""
    
    def __init__(self, model: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None, 
                 temperature: float = 0.7, max_tokens: Optional[int] = None):
        """
        Initialize the Anthropic model.
        
        Args:
            model: The Anthropic model to use (e.g., "claude-3-sonnet-20240229")
            api_key: Anthropic API key (will use ANTHROPIC_API_KEY env var if not provided)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
        """
        super().__init__(model)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens or 4096
        
        # Initialize synchronous client
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        
        # Initialize async client
        self.async_client = AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def _convert_messages_to_anthropic(self, messages: List[Message]) -> tuple[Optional[str], List[MessageParam]]:
        """Convert our Message format to Anthropic's format."""
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
                continue
                
            # Handle tool calls in assistant messages
            if msg.tool_calls:
                # Convert tool calls to Anthropic format
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                
                for tool_call in msg.tool_calls:
                    # Handle arguments which might be a JSON string
                    arguments = tool_call.function.get("arguments", {})
                    if isinstance(arguments, str):
                        import json
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    
                    content.append({
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.function.get("name", ""),
                        "input": arguments
                    })
                
                anthropic_messages.append({
                    "role": "assistant",
                    "content": content
                })
            elif msg.tool_call_id:
                # This is a tool response
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content or ""
                    }]
                })
            else:
                # Regular message
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content or ""
                })
        
        return system_message, anthropic_messages

    def _convert_tools_to_anthropic(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Convert tool definitions to Anthropic format."""
        if not tools:
            return None
            
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {})
                })
        
        return anthropic_tools

    def _convert_anthropic_response(self, response: AnthropicMessage, latency: float) -> LlmResponse:
        """Convert Anthropic response to our LlmResponse format."""
        content = ""
        tool_calls = []
        
        for content_block in response.content:
            if content_block.type == "text":
                content += content_block.text
            elif content_block.type == "tool_use":
                # Convert arguments back to JSON string format for consistency
                import json
                arguments_str = json.dumps(content_block.input) if content_block.input else "{}"
                
                tool_calls.append(ToolCall(
                    id=content_block.id,
                    type="function",
                    function={
                        "name": content_block.name,
                        "arguments": arguments_str
                    }
                ))
        
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }
        
        return LlmResponse(
            content=content if content else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=response.stop_reason or "stop",
            usage=usage,
            latency_seconds=latency
        )
    
    def model_name(self) -> str:
        return "claude"

    def complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                tool_choice: Optional[Union[str, Dict[str, Any]]] = None, **kwargs) -> LlmResponse:
        """Synchronous completion using Anthropic Claude."""
        start_time = time.time()
        
        # Create trace for this completion
        trace = self._create_trace("anthropic_completion")
        formatted_messages = self._format_messages(messages)
        generation = self._trace_generation(
            trace, "anthropic_generate", self.model, formatted_messages, 
            self.temperature, self.max_tokens, tools, tool_choice
        )
        
        try:
            system_message, anthropic_messages = self._convert_messages_to_anthropic(messages)
            anthropic_tools = self._convert_tools_to_anthropic(tools)
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs
            }
            
            if system_message:
                request_params["system"] = system_message
                
            if anthropic_tools:
                request_params["tools"] = anthropic_tools
            
            # Make the API call
            response = self.client.messages.create(**request_params)
            
            latency = time.time() - start_time
            llm_response = self._convert_anthropic_response(response, latency)
            
            # End trace with results
            self._trace_end(
                trace, generation, llm_response.content or "",
                llm_response.get_prompt_tokens(), llm_response.get_completion_tokens(),
                llm_response.get_total_tokens()
            )
            
            # Debug logging
            self._debug_print_llm_response(llm_response)
            
            return llm_response
            
        except Exception as e:
            #console_logger.log(f"Error in Anthropic completion: {str(e)}", "error")
            print(f"Error in Anthropic completion: {str(e)}", "error")
            raise

    async def async_complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                           tool_choice: Optional[Union[str, Dict[str, Any]]] = None, **kwargs) -> LlmResponse:
        """Async completion using Anthropic Claude."""
        start_time = time.time()
        
        # Create trace for this completion
        trace = self._create_trace("anthropic_async_completion")
        formatted_messages = self._format_messages(messages)
        generation = self._trace_generation(
            trace, "anthropic_async_generate", self.model, formatted_messages, 
            self.temperature, self.max_tokens, tools, tool_choice
        )
        
        try:
            system_message, anthropic_messages = self._convert_messages_to_anthropic(messages)
            anthropic_tools = self._convert_tools_to_anthropic(tools)
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs
            }
            
            if system_message:
                request_params["system"] = system_message
                
            if anthropic_tools:
                request_params["tools"] = anthropic_tools
            
            # Make the async API call
            response = await self.async_client.messages.create(**request_params)
            
            latency = time.time() - start_time
            llm_response = self._convert_anthropic_response(response, latency)
            
            # End trace with results
            self._trace_end(
                trace, generation, llm_response.content or "",
                llm_response.get_prompt_tokens(), llm_response.get_completion_tokens(),
                llm_response.get_total_tokens()
            )
            
            # Debug logging
            self._debug_print_llm_response(llm_response)
            
            return llm_response
            
        except Exception as e:
            console_logger.log(f"Error in Anthropic async completion: {str(e)}", "error")
            raise

    def stream_complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                       tool_choice: Optional[Union[str, Dict[str, Any]]] = None, **kwargs):
        """Streaming completion using Anthropic Claude."""
        try:
            system_message, anthropic_messages = self._convert_messages_to_anthropic(messages)
            anthropic_tools = self._convert_tools_to_anthropic(tools)
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": True,
                **kwargs
            }
            
            if system_message:
                request_params["system"] = system_message
                
            if anthropic_tools:
                request_params["tools"] = anthropic_tools
            
            # Return the streaming response
            return self.client.messages.create(**request_params)
            
        except Exception as e:
            console_logger.log(f"Error in Anthropic streaming: {str(e)}", "error")
            raise


# Example usage and initialization
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Initialize the model
    anthropic_model = AnthropicModel(
        model="claude-3-sonnet-20240229",
        temperature=0.7
    )
    
    # Test basic completion
    messages = [
        Message(role="user", content="What is the capital of France?")
    ]
    
    try:
        response = anthropic_model.complete(messages)
        print(f"Response: {response.content}")
        print(f"Usage: {response.usage}")
        print(f"Latency: {response.latency_seconds}s")
        
        # Test async completion
        async def test_async():
            async_response = await anthropic_model.async_complete(messages)
            print(f"Async Response: {async_response.content}")
            
            # Test batch completion
            batch_requests = [
                [Message(role="user", content="What is 2+2?")],
                [Message(role="user", content="What is the largest planet?")],
                [Message(role="user", content="Who wrote Romeo and Juliet?")]
            ]
            
            batch_responses = await anthropic_model.batch_complete(batch_requests)
            for i, resp in enumerate(batch_responses):
                print(f"Batch response {i+1}: {resp.content}")
        
        # Run async test
        asyncio.run(test_async())
        
    except Exception as e:
        print(f"Error: {e}")