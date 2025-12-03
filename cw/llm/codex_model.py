from typing import List, Optional, Dict, Any, Union
import time
import asyncio
import os
from dotenv import load_dotenv

from litellm.llms import base
from openai import AzureOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from openai import OpenAI, AsyncOpenAI

from cw.util.data_logger import get_data_logger
from cw.llm.llm_model import LlmModel, Message, LlmResponse
from cw.llm.llm_common import ToolCall
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from console_logger import console_logger


"""
 Azure SDK: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning?
 Responses API: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/responses?tabs=python-key

"""
class CodexModel(LlmModel):
    """Azure OpenAI implementation of the LlmModel interface using the official Azure OpenAI Python SDK."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
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
        self.model = "gpt-5"
        super().__init__(self.model)
        self.temperature = temperature
        self.max_tokens = max_tokens or 4096
    
    
        # Get credentials from environment or parameters
        api_key = api_key or os.getenv("GPT5_AZURE_API_KEY")
        base_url = base_url or os.getenv("GPT5_AZURE_BASE_URL")
        
        if not api_key:
            raise ValueError("Azure OpenAI API key is required. Set CODEX_AZURE_API_KEY environment variable or pass api_key parameter.")
        if not base_url:
            raise ValueError("Azure OpenAI endpoint is required. Set CODEX_AZURE_BASE_URL environment variable or pass azure_endpoint parameter.")
        
        # Initialize synchronous client
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # Initialize async client
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )


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

    def _convert_response(self, response: Any, latency: float) -> LlmResponse:
        """Convert OpenAI Responses API response to our LlmResponse format."""
        # The Responses API returns a different structure than Chat Completions
        # Extract the relevant fields based on the response structure

        # Handle both response formats
        if hasattr(response, 'choices'):
            choice = response.choices[0]
            message = choice.message

            content = message.content if hasattr(message, 'content') else None
            tool_calls = []

            # Handle tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
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
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            finish_reason = choice.finish_reason if hasattr(choice, 'finish_reason') else "stop"
        else:
            # Handle direct response format
            content = getattr(response, 'content', None) or getattr(response, 'text', None)
            tool_calls = []
            usage = {}
            finish_reason = "stop"

        return LlmResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=finish_reason,
            usage=usage,
            latency_seconds=latency
        )

    def model_name(self) -> str:
        return "openai_codex"

    def complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                tool_choice: Optional[Union[str, Dict[str, Any]]] = None, trace_name: Optional[str] = None, **kwargs) -> LlmResponse:
        """Synchronous completion using OpenAI Responses API."""
        start_time = time.time()

        try:
            openai_messages = self._convert_messages_to_openai(messages)
            openai_tools = self._convert_tools_to_openai(tools)
            formatted_messages = self._format_messages(messages)

            # Create Langfuse trace
            trace = self._create_trace(name=trace_name or "oai_codex_completion")

            # Create Langfuse generation
            generation = self._trace_generation(
                trace=trace,
                name="openai_codex_completion",
                model=self.model,
                formatted_messages=formatted_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=tools,
                tool_choice=tool_choice
            )

            # Prepare request parameters for Responses API
            # The Responses API uses similar parameters to Chat Completions
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature,
                "max_completion_tokens": self.max_tokens,  # Responses API uses max_completion_tokens
                **kwargs
            }

            if openai_tools:
                request_params["tools"] = openai_tools

            if tool_choice:
                request_params["tool_choice"] = tool_choice

            # Make the API call using the Responses endpoint
            # Note: The actual endpoint is chat.completions.create with the Responses API
            # OpenAI's SDK handles the routing based on the parameters
            response = self.client.chat.completions.create(**request_params)

            # Make the API call using the Responses endpoint
            # GPT-5-Codex requires the responses.create API instead of chat.completions
            #response = self.client.responses.create(**request_params)

            latency = time.time() - start_time
            llm_response = self._convert_response(response, latency)

            # Update Langfuse trace with results
            self._trace_end(
                trace=trace,
                generation=generation,
                output=llm_response.content or "",
                prompt_tokens=llm_response.get_prompt_tokens(),
                completion_tokens=llm_response.get_completion_tokens(),
                total_tokens=llm_response.get_total_tokens()
            )

            # Debug logging
            self._debug_print_llm_response(llm_response)
            data_logger = get_data_logger()
            data_logger.log_stats(self.model_name(), prompt_tokens=llm_response.get_prompt_tokens(),
                 completion_tokens=llm_response.get_completion_tokens(), latency_seconds=llm_response.get_latency_seconds(),
                  operation="tool_call" if tool_choice else "completion")

            return llm_response

        except Exception as e:
            console_logger.log(f"Error in OpenAI Responses completion: {str(e)}", "error")
            raise

    async def async_complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                           tool_choice: Optional[Union[str, Dict[str, Any]]] = None, trace_name: Optional[str] = None, **kwargs) -> LlmResponse:
        """Async completion using OpenAI Responses API."""
        start_time = time.time()

        try:
            openai_messages = self._convert_messages_to_openai(messages)
            openai_tools = self._convert_tools_to_openai(tools)
            formatted_messages = self._format_messages(messages)

            # Create Langfuse trace
            trace = self._create_trace(name=trace_name or "oai_responses_async_completion")

            # Create Langfuse generation
            generation = self._trace_generation(
                trace=trace,
                name="openai_responses_async_completion",
                model=self.model,
                formatted_messages=formatted_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=tools,
                tool_choice=tool_choice
            )

            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature,
                "max_completion_tokens": self.max_tokens,
                **kwargs
            }

            if openai_tools:
                request_params["tools"] = openai_tools

            if tool_choice:
                request_params["tool_choice"] = tool_choice

            # Make the async API call
            response = await self.async_client.chat.completions.create(**request_params)
            # Make the async API call using the Responses endpoint
            # GPT-5-Codex requires the responses.create API instead of chat.completions
            #response = await self.async_client.responses.create(**request_params)

            latency = time.time() - start_time
            llm_response = self._convert_response(response, latency)

            # Update Langfuse trace with results
            self._trace_end(
                trace=trace,
                generation=generation,
                output=llm_response.content or "",
                prompt_tokens=llm_response.get_prompt_tokens(),
                completion_tokens=llm_response.get_completion_tokens(),
                total_tokens=llm_response.get_total_tokens()
            )

            # Debug logging
            self._debug_print_llm_response(llm_response)

            return llm_response

        except Exception as e:
            console_logger.log(f"Error in OpenAI Responses async completion: {str(e)}", "error")
            raise

    def stream_complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
                       tool_choice: Optional[Union[str, Dict[str, Any]]] = None, **kwargs):
        """Streaming completion using OpenAI Responses API."""
        try:
            openai_messages = self._convert_messages_to_openai(messages)
            openai_tools = self._convert_tools_to_openai(tools)

            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature,
                "max_completion_tokens": self.max_tokens,
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
            console_logger.log(f"Error in OpenAI Responses streaming: {str(e)}", "error")
            raise



# Example usage and initialization
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Initialize the model
    azure_model = CodexModel(
        model="gpt-5",
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
