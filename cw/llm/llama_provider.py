from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
import time
import asyncio
import os
import json
from dotenv import load_dotenv

import groq
from groq import Groq, AsyncGroq
from groq.types.chat import ChatCompletion

from llm.llm_model import Message, LlmResponse
from cw.console_logger import console_logger
from llm.llm_common import ToolCall


class LlamaProvider(ABC):
    """Abstract base class for Llama model providers (Groq, TogetherAI, Azure, etc.)"""
    
    def __init__(self, model: str, api_key: Optional[str] = None, 
                 temperature: float = 0.7, max_tokens: Optional[int] = None):
        """
        Initialize the Llama provider.
        
        Args:
            model: The specific Llama model to use
            api_key: API key for the provider
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens or 4096
        self.api_key = api_key

    @abstractmethod
    def _convert_messages_to_provider_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert our Message format to the provider's expected format."""
        pass

    @abstractmethod
    def _convert_tools_to_provider_format(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Convert tool definitions to the provider's format."""
        pass

    @abstractmethod
    def _convert_provider_response(self, response: Any, latency: float) -> LlmResponse:
        """Convert provider response to our LlmResponse format."""
        pass

    @abstractmethod
    def complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                tool_choice: Optional[Union[str, Dict[str, Any]]] = None, **kwargs) -> LlmResponse:
        """Synchronous completion call to the provider."""
        pass

    @abstractmethod
    async def async_complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                           tool_choice: Optional[Union[str, Dict[str, Any]]] = None, **kwargs) -> LlmResponse:
        """Async completion call to the provider."""
        pass

    @abstractmethod
    def stream_complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                       tool_choice: Optional[Union[str, Dict[str, Any]]] = None, **kwargs):
        """Streaming completion call to the provider."""
        pass


class GroqLlamaProvider(LlamaProvider):
    """Groq implementation of LlamaProvider for Llama models."""
    
    def __init__(self, model: str = "meta-llama/llama-3.1-70b-versatile", 
                 api_key: Optional[str] = None, temperature: float = 0.7, 
                 max_tokens: Optional[int] = None):
        """
        Initialize the Groq Llama provider.
        
        Args:
            model: The Groq Llama model to use (default: meta-llama/llama-3.1-70b-versatile)
            api_key: Groq API key (will use GROQ_API_KEY env var if not provided)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
        """
        super().__init__(model, api_key, temperature, max_tokens)
        
        # Initialize Groq clients
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        elif os.getenv("GROQ_API_KEY"):
            client_kwargs["api_key"] = os.getenv("GROQ_API_KEY")
        
        self.client = Groq(**client_kwargs)
        self.async_client = AsyncGroq(**client_kwargs)

    def _convert_messages_to_provider_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert our Message format to Groq's format."""
        groq_messages = []
        
        for msg in messages:
            groq_msg = {"role": msg.role}
            
            # Handle content
            if msg.content:
                groq_msg["content"] = msg.content
            
            # Handle tool calls in assistant messages
            if msg.tool_calls:
                groq_msg["tool_calls"] = []
                for tool_call in msg.tool_calls:
                    groq_msg["tool_calls"].append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.get("name", ""),
                            "arguments": tool_call.function.get("arguments", "{}")
                        }
                    })
            
            # Handle tool call responses
            if msg.tool_call_id:
                groq_msg["tool_call_id"] = msg.tool_call_id
            
            groq_messages.append(groq_msg)
        
        return groq_messages

    def _convert_tools_to_provider_format(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Convert tool definitions to Groq format."""
        if not tools:
            return None
            
        groq_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                groq_tools.append(tool)  # Groq format is OpenAI-compatible
        
        return groq_tools

    def _convert_provider_response(self, response: ChatCompletion, latency: float) -> LlmResponse:
        """Convert Groq response to our LlmResponse format."""
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
        """Synchronous completion using Groq."""
        start_time = time.time()
        
        try:
            groq_messages = self._convert_messages_to_provider_format(messages)
            groq_tools = self._convert_tools_to_provider_format(tools)
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": groq_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **kwargs
            }
            
            if groq_tools:
                request_params["tools"] = groq_tools
                
            if tool_choice:
                request_params["tool_choice"] = tool_choice
            
            # Make the API call
            response = self.client.chat.completions.create(**request_params)
            
            latency = time.time() - start_time
            llm_response = self._convert_provider_response(response, latency)
            
            return llm_response
            
        except Exception as e:
            console_logger.log(f"Error in Groq completion: {str(e)}", "error")
            raise

    async def async_complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                           tool_choice: Optional[Union[str, Dict[str, Any]]] = None, **kwargs) -> LlmResponse:
        """Async completion using Groq."""
        start_time = time.time()
        
        try:
            groq_messages = self._convert_messages_to_provider_format(messages)
            groq_tools = self._convert_tools_to_provider_format(tools)
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": groq_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **kwargs
            }
            
            if groq_tools:
                request_params["tools"] = groq_tools
                
            if tool_choice:
                request_params["tool_choice"] = tool_choice
            
            # Make the async API call
            response = await self.async_client.chat.completions.create(**request_params)
            
            latency = time.time() - start_time
            llm_response = self._convert_provider_response(response, latency)
            
            return llm_response
            
        except Exception as e:
            console_logger.log(f"Error in Groq async completion: {str(e)}", "error")
            raise

    def stream_complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                       tool_choice: Optional[Union[str, Dict[str, Any]]] = None, **kwargs):
        """Streaming completion using Groq."""
        try:
            groq_messages = self._convert_messages_to_provider_format(messages)
            groq_tools = self._convert_tools_to_provider_format(tools)
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": groq_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True,
                **kwargs
            }
            
            if groq_tools:
                request_params["tools"] = groq_tools
                
            if tool_choice:
                request_params["tool_choice"] = tool_choice
            
            # Return the streaming response
            return self.client.chat.completions.create(**request_params)
            
        except Exception as e:
            console_logger.log(f"Error in Groq streaming: {str(e)}", "error")
            raise


# Example usage and factory function
def create_groq_llama_provider(model: str = "meta-llama/llama-3.1-70b-versatile", 
                              api_key: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None) -> GroqLlamaProvider:
    """Factory function to create a GroqLlamaProvider."""
    return GroqLlamaProvider(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )


# For future providers
class TogetherAILlamaProvider(LlamaProvider):
    """Placeholder for TogetherAI Llama provider - to be implemented later."""
    pass


class AzureLlamaProvider(LlamaProvider):
    """Placeholder for Microsoft Azure Llama provider - to be implemented later."""
    pass


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Test the Groq provider
    provider = create_groq_llama_provider()
    
    # Test basic completion
    messages = [
        Message(role="user", content="What is the capital of France?")
    ]
    
    try:
        response = provider.complete(messages)
        print(f"Response: {response.content}")
        print(f"Usage: {response.usage}")
        print(f"Latency: {response.latency_seconds}s")
        
        # Test async completion
        async def test_async():
            async_response = await provider.async_complete(messages)
            print(f"Async Response: {async_response.content}")
        
        # Run async test
        asyncio.run(test_async())
        
    except Exception as e:
        print(f"Error: {e}")