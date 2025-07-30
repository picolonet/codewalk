from typing import Dict, Any, List, Optional, Callable
from llm.llm_model import LlmModel, Message, ToolCall, ToolCallResponse, format_messages
import json
import inspect
from pydantic import BaseModel
from util.livepanel import LivePanel
from console_logger import console_logger

import os

IGNORE_FILE_LIST = [".git", ".env", ".github", ".claude", ".venv_1"]

def filter_list(input_list, ignore_list):
    return [item for item in input_list if item not in ignore_list]


class CompletionResult(BaseModel):
    """Result of a completion with tools. Includes the full conversation and the current conversation."""
    has_tool_calls: bool
    full_conversation: List[Message]
    current_result: List[Message]
    last_response: Message # The final message

class ToolCaller:
    """Handles tool registration and execution for LLM interactions."""
    
    def __init__(self, llm_model: LlmModel):
        self.llm_model = llm_model
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tool_functions: Dict[str, Callable] = {}

    # def start_debugpanel(self):
    #     self.debug_panel = LivePanel("Debug messages")
    #     self.debug_panel.start()

    # def post_debugpanel(self, title: str, messages: List[Message]):
    #     formatted_message = format_messages(messages)
    #     self.post_debugpanel_formatted(title, formatted_message)

    # def post_debugpanel_formatted(self, title: str, message: str):
    #     self.debug_panel.add_message(f"{title}\n {message}")

    # def stop_debugpanel(self):
    #     self.debug_panel.stop()
    
    def post_json(self, title:str, messages: List[Message], type: str):
        # Convert Pydantic models to dicts for JSON rendering
        message_dicts = [m.model_dump(exclude_none=True) for m in messages]
        console_logger.log_json(message_dicts, title=title, type=type)

    
    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], 
                     function: Callable) -> None:
        """Register a tool with its schema and implementation function."""
        tool_schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        self.tools[name] = tool_schema
        self.tool_functions[name] = function
    
    def register_tool_from_function(self, function: Callable, description: Optional[str] = None) -> None:
        """Register a tool automatically from a function's signature and docstring."""
        name = function.__name__
        
        # Use provided description or extract from docstring
        if description is None:
            description = function.__doc__ or f"Function {name}" or ""
        
        # Extract parameters from function signature
        sig = inspect.signature(function)
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            param_type = "string"  # Default type
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
                elif param.annotation == dict:
                    param_type = "object"
            
            parameters["properties"][param_name] = {"type": param_type}
            
            # Add to required if no default value
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)
        
        self.register_tool(name, description, parameters, function)
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get all registered tool schemas for LLM."""
        return list(self.tools.values())
    
    def execute_tool(self, tool_call: ToolCall) -> ToolCallResponse:
        """Execute a tool call and return the response."""
        function_name = tool_call.function.get("name")
        
        if function_name not in self.tool_functions:
            return ToolCallResponse(
                id=f"resp_{tool_call.id}",
                content=f"Error: Tool '{function_name}' not found",
                tool_call_id=tool_call.id
            )
        
        try:
            # Parse arguments
            arguments_str = tool_call.function.get("arguments", "{}")
            arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
            
            # Execute the function
            result = self.tool_functions[function_name](**arguments)
            
            # Convert result to string if needed
            if not isinstance(result, str):
                result = str(result)
            
            return ToolCallResponse(
                id=f"resp_{tool_call.id}",
                content=result,
                tool_call_id=tool_call.id
            )
            
        except Exception as e:
            return ToolCallResponse(
                id=f"resp_{tool_call.id}",
                content=f"Error executing {function_name}: {str(e)}",
                tool_call_id=tool_call.id
            )
    
    def complete_with_tools(self, messages: List[Message], tool_choice: Optional[str] = "auto",
                           max_iterations: int = 5) -> List[Message]:
        """Complete a conversation with automatic tool execution."""
        conversation = messages.copy()
        
        for iteration in range(max_iterations):
            # Get response from LLM
            response = self.llm_model.complete(
                messages=conversation,
                tools=self.get_tool_schemas(),
                tool_choice=tool_choice
            )
            
            # Add assistant message to conversation
            assistant_message = Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls
            )
            conversation.append(assistant_message)
            
            # If no tool calls, we're done
            if not response.tool_calls:
                break
            
            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_response = self.execute_tool(tool_call)
                
                # Add tool response to conversation
                tool_message = Message(
                    role="tool",
                    content=tool_response.content,
                    tool_call_id=tool_response.tool_call_id
                )
                conversation.append(tool_message)
        
        return conversation

    def full_completion_with_tools(self, messages: List[Message], tool_choice: Optional[str] = "auto",
                           max_iterations: int = 50 ) -> CompletionResult:
        """Complete a conversation with automatic tool execution."""

        # Debug live panel to print each round of messages
        # self.start_debugpanel()
        # self.post_debugpanel("New Query:", messages=messages)
        self.post_json("New Query", messages=messages, type="user")

        full_conversation = messages.copy()
        current_result: List[Message] = []
        has_tool_calls = False    
        
        for iteration in range(max_iterations):
            # Get response from LLM
            last_response = self.llm_model.complete(messages=full_conversation, tools=self.get_tool_schemas(),
                tool_choice=tool_choice)
            print(f"Latency (sec) = {last_response.latency_seconds}")
            
            # Add assistant message to conversation
            assistant_message = Message(
                role = "assistant",
                content = last_response.content,
                tool_calls = last_response.tool_calls
            )

            # Post to log message
            self.post_json("Response from LLM:", [assistant_message], type="from_llm")

            full_conversation.append(assistant_message)
            current_result.append(assistant_message)
            
            
            # If no tool calls, we're done
            if not last_response.tool_calls:
                break
            has_tool_calls = True
            
            # Execute each tool call
            for tool_call in last_response.tool_calls:
                tool_response = self.execute_tool(tool_call)
                
                # Add tool response to conversation
                tool_message = Message(
                    role="tool",
                    content=tool_response.content,
                    tool_call_id=tool_response.tool_call_id
                )
                # log message: Tool call
                self.post_json("Tool call output to LLM:", [tool_message], type = "to_llm")
                full_conversation.append(tool_message)
                current_result.append(tool_message)
        
        # self.stop_debugpanel()
        return CompletionResult(has_tool_calls=has_tool_calls, full_conversation=full_conversation,
                                 current_result=current_result, last_response=assistant_message)



# Example usage functions that could be registered as tools
def get_file_contents(file_path: str) -> str:
    """Read and return the contents of a file."""
    if os.path.getsize(file_path) == 0:
        return "empty file"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def list_directory(directory_path: str) -> str:
    """List files and directories in the given path."""
    actual_result = _list_directory_internal(directory_path)
    result = f"Result of listing files in {directory_path}:\n<result>\n{actual_result}\n<\result>"
    return result

def _list_directory_internal(directory_path: str) -> str:
    """List files and directories in the given path."""
    import os
    try:
        items = os.listdir(directory_path)
        items = filter_list(items, IGNORE_FILE_LIST)
        return "\n".join(items)
    except Exception as e:
        return f"Error listing directory: {str(e)}"


def search_files(directory: str, pattern: str) -> str:
    """Search for files matching a pattern in a directory."""

    actual_result = _search_files_internal(directory, pattern)
    result = f"Search result for files in {directory} matching {pattern}:\n<result>\n{actual_result}\n<\result>"
    return result
    
def _search_files_internal(directory: str, pattern: str) -> str:
    """Search for files matching a pattern in a directory."""
    import os
    import fnmatch
    
    try:
        matches = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if fnmatch.fnmatch(file, pattern):
                    matches.append(os.path.join(root, file))
        
        return "\n".join(matches) if matches else "No files found matching pattern"
    except Exception as e:
        return f"Error searching files: {str(e)}"
