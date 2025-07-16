from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import litellm
from litellm import completion
from rich.console import Console
from rich.panel import Panel
import json
from litellm.types.utils import ModelResponse
from litellm.types.utils import StreamingChoices


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