from trace import Trace
from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
import asyncio
from pydantic import BaseModel, Field
import litellm
from litellm import completion
from rich.console import Console
from rich.panel import Panel
import json
from litellm.types.utils import ModelResponse
from litellm.types.utils import StreamingChoices
import os
import time
from dotenv import load_dotenv
from console_logger import console_logger
from langfuse import Langfuse


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
  
    def to_string(self) -> str:
        """Convert a Message object to a string representation."""
        result = []
        if self.content:
            result.append(self.content)
        
        if self.tool_calls:
            for tool_call in self.tool_calls:
                result.append(f"Tool Call {tool_call.id}:")
                result.append(f"Function: {tool_call.function.get('name')}")
                result.append(f"Arguments: {json.dumps(tool_call.function.get('arguments'), indent=2)}")
        
        if self.tool_call_id:
            result.append(f"Tool Call ID: {self.tool_call_id}")
            
        return "\n".join(result)


class LlmResponse(BaseModel):
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: str
    usage: Dict[str, Any]
    latency_seconds: Optional[float] = None

    def get_latency_seconds(self) -> float:
        return self.latency_seconds
    
    def get_prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)
    
    def get_completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)

    def get_total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


def format_messages(messages: List[Message]) -> List[Dict[str, Any]]:
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


class LlmModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Initialize Langfuse for tracing
        self.langfuse = Langfuse()


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

    def _debug_print_llm_response(self, llm_response: LlmResponse):
        """Pretty prints LLM response for debugging using rich."""
        console = Console()
        
        # Create a dictionary representation of the response
        response_dict = {
            "content": llm_response.content,
            "finish_reason": llm_response.finish_reason,
            "usage": llm_response.usage,
            "latency_seconds": llm_response.latency_seconds
        }
        
        if llm_response.tool_calls:
            response_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": tc.function
                }
                for tc in llm_response.tool_calls
            ]
            
        # Print as formatted JSON
        json_str = json.dumps(response_dict, indent=2)
        console_logger.log_json_panel(response_dict, title="LLM Response", type = "from_llm")
        #console.print(Panel(json_str, title="LLM Response", border_style="green"))

    def _create_trace(self, name: str):
        """Create a new trace in Langfuse v3."""
        return self.langfuse.trace(name=name)

    def _trace_generation(self, trace, name: str, model: str, formatted_messages: List[Dict[str, Any]],
        temperature: float, max_tokens: Optional[int] = None, tools: Optional[List[Dict[str, Any]]] = None, 
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None):
        """Create a generation within a trace in Langfuse v3."""
        generation = trace.generation(
            name=name,
            model=model,
            input=formatted_messages,
            metadata={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "tools": len(tools) if tools else 0,
                "tool_choice": tool_choice
            }
        )
        return generation

    def _trace_end(self, generation, output: str, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        """End a generation in Langfuse v3."""
        generation.end(
            output=output,
            usage={
                "input": prompt_tokens,
                "output": completion_tokens,
                "total": total_tokens
            }
        )


    def get_model_name(self) -> str:
        return self.model_name

    @abstractmethod
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
        pass

    @abstractmethod
    def stream_complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs):
        pass

    @abstractmethod
    async def async_complete(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None, **kwargs) -> LlmResponse:
        """Async version of complete method for parallel processing.
        
        Args:
            messages: List of Message objects representing the conversation history
            tools: Optional list of tool definitions that the model can use
            tool_choice: Optional specification of which tool the model should use
            **kwargs: Additional keyword arguments passed to the completion API

        Returns:
            LlmResponse containing the model's response
        """
        pass

    async def batch_complete(self, requests: List[List[Message]], 
                           tools: Optional[List[Dict[str, Any]]] = None,
                           tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
                           **kwargs) -> List[LlmResponse]:
        """Process multiple completion requests in parallel.
        
        Args:
            requests: List of message lists, each representing a separate completion request
            tools: Optional list of tool definitions that the model can use
            tool_choice: Optional specification of which tool the model should use
            **kwargs: Additional keyword arguments passed to the completion API
            
        Returns:
            List of LlmResponse objects corresponding to each request
        """
        tasks = [
            self.async_complete(messages, tools=tools, tool_choice=tool_choice, **kwargs)
            for messages in requests
        ]
        return await asyncio.gather(*tasks)
        