from typing import List, Optional, Dict, Any, Union
import time
import asyncio
import os
from dotenv import load_dotenv

from cw.util.data_logger import get_data_logger
from llm.llm_model import LlmModel, Message, LlmResponse
from llm.llama_provider import LlamaProvider, GroqLlamaProvider, create_groq_llama_provider
from cw.console_logger import console_logger
from llm.llm_common import ToolCall
from cw.cw_config import get_cw_config, CwConfig

class LlamaModel(LlmModel):
    """Llama model implementation that uses pluggable LlamaProvider backends."""
    
    def __init__(self, provider: Optional[LlamaProvider] = None, 
                 model: str = "meta-llama/llama-3.1-70b-versatile",
                 provider_type: str = "groq",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None):
        """
        Initialize the Llama model with a provider.
        
        Args:
            provider: Optional LlamaProvider instance. If None, will create one based on provider_type
            model: The specific Llama model to use
            provider_type: Type of provider to create if provider is None ("groq", "together", "azure")
            api_key: API key for the provider
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
        """
        super().__init__(model_name=model)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        
        # Initialize provider
        if provider:
            self.provider = provider
        else:
            self.provider = self._create_provider(provider_type, model, api_key, temperature, max_tokens)

    def _create_provider(self, provider_type: str, model: str, api_key: Optional[str],
                        temperature: float, max_tokens: Optional[int]) -> LlamaProvider:
        """Factory method to create the appropriate provider."""
        if provider_type.lower() == "groq":
            return create_groq_llama_provider(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif provider_type.lower() == "together":
            # Placeholder for future TogetherAI provider
            raise NotImplementedError("TogetherAI provider not yet implemented")
        elif provider_type.lower() == "azure":
            # Placeholder for future Azure provider
            raise NotImplementedError("Azure provider not yet implemented")
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")


    def model_name(self) -> str:
        return "llama"

    def complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                tool_choice: Optional[Union[str, Dict[str, Any]]] = None,  trace_name: Optional[str] = None, **kwargs) -> LlmResponse:
        """Synchronous completion using the configured provider."""

        # Create Langfuse trace
        formatted_messages = self._format_messages(messages)
        trace = self._create_trace(name=trace_name or "llm_completion")
        generation = self._trace_generation(trace=trace, name="litellm_completion", model=self.model, formatted_messages=formatted_messages,
            temperature=self.temperature, max_tokens=self.max_tokens, tools=tools, tool_choice=tool_choice)
        
        try:
            response = self.provider.complete(messages, tools=tools, tool_choice=tool_choice, **kwargs)
            message_content = response.content
            
            # Debug logging
            self._debug_print_llm_response(response)
            data_logger = get_data_logger()
            data_logger.log_stats(self.get_model_name(), prompt_tokens=response.get_prompt_tokens(),
                 completion_tokens=response.get_completion_tokens(), latency_seconds=response.get_latency_seconds(),
                  operation="tool_call" if tools != None else "completion")

            self._trace_end(generation=generation, output=message_content, prompt_tokens=response.get_prompt_tokens(),
            completion_tokens=response.get_completion_tokens(), total_tokens=response.get_total_tokens())
            return response
            
        except Exception as e:
            console_logger.log_text(f"Error in Llama completion: {str(e)}", "error")
            raise

    async def async_complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                           tool_choice: Optional[Union[str, Dict[str, Any]]] = None, **kwargs) -> LlmResponse:
        """Async completion using the configured provider."""
        try:
            response = await self.provider.async_complete(messages, tools=tools, tool_choice=tool_choice, **kwargs)
            
            # Debug logging
            self._debug_print_llm_response(response)
            
            return response
            
        except Exception as e:
            console_logger.log_text(f"Error in Llama async completion: {str(e)}", "error")
            raise

    def stream_complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                       tool_choice: Optional[Union[str, Dict[str, Any]]] = None, **kwargs):
        """Streaming completion using the configured provider."""
        try:
            return self.provider.stream_complete(messages, tools=tools, tool_choice=tool_choice, **kwargs)
            
        except Exception as e:
            console_logger.log_text(f"Error in Llama streaming: {str(e)}", "error")
            raise

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider."""
        return {
            "provider_type": self.provider.__class__.__name__,
            "model": self.provider.model,
            "temperature": self.provider.temperature,
            "max_tokens": self.provider.max_tokens
        }


# Convenience factory functions for different providers
def create_groq_llama_model(model: str = "meta-llama/llama-4-maverick-17b-128e-instruct",
                           api_key: Optional[str] = None,
                           temperature: float = 0.7,
                           max_tokens: Optional[int] = None) -> LlamaModel:
    """Create a LlamaModel using Groq provider."""
    cw_config = get_cw_config()
    groq_model = cw_config.get(CwConfig.GROQ_MODEL_KEY, CwConfig.GROQ_MODEL_DEFAULT)
    # if model is None:
    #     model = groq_model  

    return LlamaModel(
        provider_type="groq",
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )


def create_llama4_scout_model(api_key: Optional[str] = None,
                             temperature: float = 0.7,
                             max_tokens: Optional[int] = None) -> LlamaModel:
    """Create a LlamaModel specifically for Llama4 Scout using Groq."""
    print(f"Creating Llama4 Scout model.")
    return LlamaModel(
        provider_type="groq",
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )


# Example usage and initialization
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Test different ways to create LlamaModel
    
    # Method 1: Using default Groq provider
    llama_model = LlamaModel()
    
    # Method 2: Using convenience factory for Groq
    groq_llama = create_groq_llama_model(model="meta-llama/llama-3.1-8b-instant")
    
    # Method 3: Using specific Llama4 Scout model
    llama4_scout = create_llama4_scout_model()
    
    # Method 4: Using custom provider instance
    custom_provider = create_groq_llama_provider(model="meta-llama/llama-3.1-70b-versatile")
    custom_llama = LlamaModel(provider=custom_provider)
    
    # Test basic completion
    messages = [
        Message(role="user", content="Explain what makes Llama models unique.")
    ]
    
    try:
        print("Testing Llama4 Scout model:")
        response = llama4_scout.complete(messages)
        print(f"Response: {response.content}")
        print(f"Usage: {response.usage}")
        print(f"Latency: {response.latency_seconds}s")
        print(f"Provider info: {llama4_scout.get_provider_info()}")
        
        # Test with tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge",
                    "description": "Search for information in a knowledge base",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            },
                            "domain": {
                                "type": "string",
                                "enum": ["science", "technology", "history", "general"],
                                "description": "The domain to search in"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        tool_messages = [
            Message(role="user", content="Can you search for information about machine learning?")
        ]
        
        print("\nTesting with tools:")
        tool_response = llama4_scout.complete(tool_messages, tools=tools, tool_choice="auto")
        print(f"Tool Response: {tool_response.content}")
        if tool_response.tool_calls:
            print(f"Tool calls made: {len(tool_response.tool_calls)}")
            for i, tool_call in enumerate(tool_response.tool_calls):
                print(f"  Tool {i+1}: {tool_call.function}")
        
        # Test async completion
        async def test_async():
            print("\nTesting async completion:")
            async_response = await llama4_scout.async_complete(messages)
            print(f"Async Response: {async_response.content}")
            
            # Test batch completion
            batch_requests = [
                [Message(role="user", content="What are the benefits of open-source AI?")],
                [Message(role="user", content="How do transformer models work?")],
                [Message(role="user", content="What is the difference between LLaMA and GPT?")]
            ]
            
            print("\nTesting batch completion:")
            batch_responses = await llama4_scout.batch_complete(batch_requests)
            for i, resp in enumerate(batch_responses):
                print(f"Batch response {i+1}: {resp.content[:100]}...")
        
        # Run async test
        asyncio.run(test_async())
        
        # Test streaming
        print("\n" + "="*50)
        print("Testing streaming:")
        
        stream_messages = [
            Message(role="user", content="Tell me about the evolution of language models.")
        ]
        
        stream = llama4_scout.stream_complete(stream_messages)
        print("Streaming response: ", end="", flush=True)
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()  # New line after streaming
        
    except Exception as e:
        print(f"Error: {e}")