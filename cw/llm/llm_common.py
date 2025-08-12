from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from cw.console_logger import console_logger

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


class CompletionResult(BaseModel):
    """Result of a completion with tools. Includes the full conversation and the current conversation."""
    has_tool_calls: bool
    full_conversation: List[Message]
    current_result: List[Message]
    last_response: Message # The final message
    user_facing_result: str # The final result from the LLM to be displayed to the user



class LlmUtils:
    def __init__(self):
        pass

    @staticmethod
    def post_json(title:str, messages: List[Message], type: str):
        # Convert Pydantic models to dicts for JSON rendering
        message_dicts = [m.model_dump(exclude_none=True) for m in messages]
        console_logger.log_json_panel(message_dicts, title=title, type=type)