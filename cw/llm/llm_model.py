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
from cw.console_logger import console_logger
from langfuse import Langfuse
from cw.llm.llm_common import Message, LlmResponse
import logging
from datetime import datetime


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

    def _setup_llm_logging(self):
        """Setup LLM logging to capture errors to a file in logs/ directory"""
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Create a logger for LiteLLM errors
        self.error_logger = logging.getLogger(f"{self.model_name}_errors")
        self.error_logger.setLevel(logging.ERROR)
        
        # Create file handler for error logs
        error_log_file = f"logs/{self.model_name}_llm_errors_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(error_log_file)
        file_handler.setLevel(logging.ERROR)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger if not already added
        if not self.error_logger.handlers:
            self.error_logger.addHandler(file_handler)
        

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
        console_logger.log_json_panel(response_dict, title="LlmModel: LLM Response", type = "from_llm")
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
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None, trace_name: Optional[str] = None, **kwargs) -> LlmResponse:
        """Completes a chat conversation using the configured LLM.

        Args:
            messages: List of Message objects representing the conversation history
            tools: Optional list of tool definitions that the model can use
            tool_choice: Optional specification of which tool the model should use
            trace_name: Optional name of the trace to create in Langfuse
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
        