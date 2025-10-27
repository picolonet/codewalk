from typing import List, Optional, Dict, Any, Union
import time
import asyncio
import os
from dotenv import load_dotenv

from openai import AzureOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from llm.llm_model import LlmModel, Message, LlmResponse
from llm.llm_common import ToolCall
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from console_logger import console_logger

0
"""
 Azure SDK: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning?
 https://arune-mfpva4eo-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-5/chat/completions?api-version=2025-01-01-preview

Rate limit:  https://aka.ms/oai/quotaincrease.
"""
class AzureOpenAIModel(LlmModel):
    """Azure OpenAI implementation of the LlmModel interface using the official Azure OpenAI Python SDK."""
    
    def __init__(self, model: str = "gpt-5", api_key: Optional[str] = None, 
                 api_version: str = "2025-01-01-preview", azure_endpoint: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: Optional[int] = None):
        """
        Initialize the Azure OpenAI model.
        
        Args:
            model: The Azure OpenAI model to use (e.g., "gpt-4", "gpt-35-turbo")
            api_key: Azure OpenAI API key (will use AZURE_OPENAI_API_KEY env var if not provided)
            api_version: Azure OpenAI API version
            azure_endpoint: Azure OpenAI endpoint URL (will use AZURE_OPENAI_ENDPOINT env var if not provided)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
        """
        super().__init__(model)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens or 4096
        self.api_version = api_version
        
        # Get credentials from environment or parameters
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if not api_key:
            raise ValueError("Azure OpenAI API key is required. Set AZURE_OPENAI_API_KEY environment variable or pass api_key parameter.")
        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint is required. Set AZURE_OPENAI_ENDPOINT environment variable or pass azure_endpoint parameter.")
        
        # Initialize synchronous client
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
        
        # Initialize async client
        self.async_client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )

    """
    Response format:
    # 5️⃣ Send the tool result back to GPT-5
    response_2 = client.responses.create(
    model="gpt-5",
    input=[
        {"role": "system", "content": "You are a weather assistant."},
        {"role": "user", "content": "What's the weather in Boston?"},
        {
            "role": "assistant",
            "tool_calls": tool_calls
        },
        {
            "role": "tool",
            "tool_call_id": tool_calls[0]["id"],
            "content": tool_result
        }
    ]
    )
    """
    def _convert_messages_to_openai(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert our Message format to OpenAI's format."""
        openai_messages = []
        
        for msg in messages:
            # Handle tool calls in assistant messages
            if msg.tool_calls:
                oai_msg = {"role": "assistant"}
                # Convert tool calls to OpenAI format
                #content = []
                if msg.content:
                    oai_msg["content"] = msg.content
                    #content.append({"type": "text", "text": msg.content})
                
                oai_msg["tool_calls"] = []
                for tool_call in msg.tool_calls:
                    # Handle arguments which might be a JSON string
                    arguments = tool_call.function.get("arguments", {})
                    if isinstance(arguments, str):
                        import json
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                               arguments = {}

                    # oai_msg["tool_calls"].append({
                    #     "type": "tool_call",
                    #     "tool_call": {
                    #         "id": tool_call.id,
                    #         "type": "function",
                    #         "function": {
                    #             "name": tool_call.function.get("name", ""),
                    #             "arguments": json.dumps(arguments) if isinstance(arguments, dict) else str(arguments)
                    #         }
                    #     }
                    # })
                    oai_msg["tool_calls"].append({
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.get("name", ""),
                                "arguments": json.dumps(arguments) if isinstance(arguments, dict) else str(arguments)
                            }
                    })
                
                openai_messages.append(oai_msg)
            elif msg.tool_call_id:
                # This is a tool response
                openai_messages.append({
                    "role": "tool",
                    "content": msg.content or "",
                    "tool_call_id": msg.tool_call_id
                })
            else:
                # Regular message
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content or ""
                })
        
        return openai_messages

    def _convert_tools_to_openai(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Convert tool definitions to OpenAI format."""
        if not tools:
            return None
            
        openai_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {})
                    }
                })
        
        return openai_tools

    def _convert_openai_response(self, response: ChatCompletion, latency: float) -> LlmResponse:
        """Convert OpenAI response to our LlmResponse format."""
        message = response.choices[0].message
        content = message.content or ""
        tool_calls = []
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                # Handle arguments which might be a JSON string
                arguments = tool_call.function.arguments
                if isinstance(arguments, str):
                    import json
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                
                tool_calls.append(ToolCall(
                    id=tool_call.id,
                    type="function",
                    function={
                        "name": tool_call.function.name,
                        "arguments": json.dumps(arguments) if isinstance(arguments, dict) else str(arguments)
                    }
                ))
        
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        return LlmResponse(
            content=content if content else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=response.choices[0].finish_reason or "stop",
            usage=usage,
            latency_seconds=latency
        )
    
    def model_name(self) -> str:
        return "azure-openai"

    
    def complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                tool_choice: Optional[Union[str, Dict[str, Any]]] = None, trace_name: Optional[str] = None,
                **kwargs) -> LlmResponse:
        """Synchronous completion using Azure OpenAI."""
        start_time = time.time()
        
        # Create trace for this completion
        trace = self._create_trace("azure_openai_completion")
        formatted_messages = self._format_messages(messages)
        trace_name = trace_name or "azure_openai_generate"
        generation = self._trace_generation(
            trace, trace_name, self.model, formatted_messages, 
            self.temperature, self.max_tokens, tools, tool_choice
        )
        
        try:
            openai_messages = self._convert_messages_to_openai(messages)
            openai_tools = self._convert_tools_to_openai(tools)
            
            # Prepare request parameters
            # GPT-5 openai does not support max_tokens and temperature of anything other than 1.0
            request_params = {
                "model": self.model,
                "messages": openai_messages,
            #    "max_tokens": self.max_tokens,
             #   "temperature": self.temperature,
                **kwargs
            }

            console_logger.log_json_panel(request_params, title="REQUEST PARAMS")
            
            if openai_tools:
                request_params["tools"] = openai_tools
                if tool_choice:
                    request_params["tool_choice"] = tool_choice
            
            # Make the API call
            response = self.client.chat.completions.create(**request_params)
            
            latency = time.time() - start_time
            llm_response = self._convert_openai_response(response, latency)
            
            # End trace with results
            self._trace_end(
                generation, llm_response.content or "",
                llm_response.get_prompt_tokens(), llm_response.get_completion_tokens(),
                llm_response.get_total_tokens()
            )
            
            # Debug logging
            self._debug_print_llm_response(llm_response)
            
            return llm_response
            
        except Exception as e:
            console_logger.log_text(f"Error in Azure OpenAI completion: {str(e)}", "error")
            raise

    async def async_complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                           tool_choice: Optional[Union[str, Dict[str, Any]]] = None, **kwargs) -> LlmResponse:
        """Async completion using Azure OpenAI."""
        start_time = time.time()
        
        # Create trace for this completion
        trace = self._create_trace("azure_openai_async_completion")
        formatted_messages = self._format_messages(messages)
        generation = self._trace_generation(
            trace, "azure_openai_async_generate", self.model, formatted_messages, 
            self.temperature, self.max_tokens, tools, tool_choice
        )
        
        try:
            openai_messages = self._convert_messages_to_openai(messages)
            openai_tools = self._convert_tools_to_openai(tools)
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
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
            
            # End trace with results
            self._trace_end(
                generation, llm_response.content or "",
                llm_response.get_prompt_tokens(), llm_response.get_completion_tokens(),
                llm_response.get_total_tokens()
            )
            
            # Debug logging
            self._debug_print_llm_response(llm_response)
            
            return llm_response
            
        except Exception as e:
            console_logger.log_text(f"Error in Azure OpenAI async completion: {str(e)}", "error")
            raise

    def stream_complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                       tool_choice: Optional[Union[str, Dict[str, Any]]] = None, **kwargs):
        """Streaming completion using Azure OpenAI."""
        try:
            openai_messages = self._convert_messages_to_openai(messages)
            openai_tools = self._convert_tools_to_openai(tools)
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
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
            console_logger.log_text(f"Error in Azure OpenAI streaming: {str(e)}", "error")
            raise


# Example usage and initialization
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Initialize the model
    azure_model = AzureOpenAIModel(
        model="gpt-5-codex",
        temperature=0.7
    )
    
    # Test basic completion
    messages = [
        Message(role="user", content="What is the capital of France?")
    ]
    
    try:
        response = azure_model.complete(messages)
        print(f"Response: {response.content}")
        print(f"Usage: {response.usage}")
        print(f"Latency: {response.latency_seconds}s")
        
        # Test async completion
        async def test_async():
            async_response = await azure_model.async_complete(messages)
            print(f"Async Response: {async_response.content}")
            
            # Test batch completion
            batch_requests = [
                [Message(role="user", content="What is 2+2?")],
                [Message(role="user", content="What is the largest planet?")],
                [Message(role="user", content="Who wrote Romeo and Juliet?")]
            ]
            
            batch_responses = await azure_model.batch_complete(batch_requests)
            for i, resp in enumerate(batch_responses):
                print(f"Batch response {i+1}: {resp.content}")
        
        # Run async test
        asyncio.run(test_async())
        
    except Exception as e:
        print(f"Error: {e}")
