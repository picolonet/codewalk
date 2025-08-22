import inspect
import json
from typing import Callable, List, Optional, Dict, Any

from aiohttp.payload import TOO_LARGE_BYTES_BODY
from cw.console_logger import console_logger
from cw.llm.llm_router import llm_router
from cw.util.data_logger import get_data_logger
from tool_caller import CompletionResult, ToolCaller, get_file_contents, list_directory, search_files
from enum import Enum
import xml.etree.ElementTree as ET
from datetime import datetime
from pydantic import BaseModel
from cw_prompts import cw_analyze_file_prompt, cw_analyze_v0, cwcc_system_prompt, cw_todo_write_tool_description, TODO_MODIFICATION_RESPONSE_MSG
from cw.util.cw_common import get_env
from llm.llm_completion import LlmCompletion
from llm.llm_common import Message, CompletionResult

class TodoStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TodoPriority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TodoItem(BaseModel):
    """Represents a single todo item with id, content, status, and priority."""
    id: str
    content: str
    status: TodoStatus  # pending, in_progress, completed
    priority: TodoPriority  # high, medium, low
    
    def to_dict(self) -> Dict[str, str]:
        """Convert TodoItem to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status.value,
            "priority": self.priority.value
        }
        

class CwTask:
    def __init__(self, user_query: str, code_base_path: str, task_system_prompt: str = "", operation_tag: str = "completion"):
        self.user_query = user_query
        self.code_base_path = code_base_path
        self.memory: List[Message] = []
        self.todos: List[TodoItem] = []
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tool_functions: Dict[str, Callable] = {}
        self.operation_tag = operation_tag
        self.tool_caller = ToolCaller()
        self.tool_caller.register_basic_tools()
        self.tool_caller.register_tool_from_function(self.todo_write, cw_todo_write_tool_description())
        system_prompt = cwcc_system_prompt(get_env()) if task_system_prompt == "" else task_system_prompt
        self.memory.append(Message(role="system", content=system_prompt))
        # self.memory.append(Message(role="system", content=cwcc_system_prompt(get_env())))

        # self.register_tool_from_function(get_file_contents)
        # self.register_tool_from_function(list_directory)
        # self.register_tool_from_function(search_files)
        # self.register_tool_from_function(self.todo_write, cw_todo_write_tool_description())
        # self.memory.append(Message(role="system", content=cwcc_system_prompt(get_env())))

    def get_tool_caller(self) -> ToolCaller:
        """Get the tool caller for the task to register additional tools."""
        return self.tool_caller

    def conversation_history_preprocessor(self, messages: List[Message]) -> List[Message]:
        """Pre-process the conversation history before the LLM is called."""
        if self.has_todos():
            messages.append(self.todo_summary_message())
            console_logger.log_text(f"Appending TODOs: {self.todo_to_json()}")
        return messages

    def are_todos_complete(self) -> bool:
        """Check if all todos are completed."""
        return all(todo.status == TodoStatus.COMPLETED.value for todo in self.todos)

    def has_todos(self) -> bool:
        """Check if there are any todos."""
        return len(self.todos) > 0


    def run(self, query: str):
        query_message = Message(role="user", content=query)
        llm_completion = LlmCompletion(self.memory, self.tool_caller)
        llm_completion.register_conversation_history_preprocessor(self.conversation_history_preprocessor)
        return llm_completion.complete(query_message, trace_name=self.operation_tag, operation_tag=self.operation_tag)

    def todo_write(self, todos: List[Dict[str, Any]]) -> str:
        """Tool function to write a todo list.
        
        Args:
            todos: List of todo dictionaries, each containing:
                - id: Unique identifier for the todo
                - content: Description of the todo task
                - status: Current status (pending, in_progress, completed)
                - priority: Task priority (high, medium, low)
                
        Returns:
            str: Success message indicating todos were processed
        
        Example input format:
            [
                {
                    "id": "1",
                    "content": "Explore project structure and entry points", 
                    "status": "pending",
                    "priority": "high"
                },
                {
                    "id": "2",
                    "content": "Identify main application files",
                    "status": "in_progress", 
                    "priority": "medium"
                }
            ]
        """
        console_logger.log_text(f"TOOL CALL Updating todos: {todos}")
        return self._todo_write_internal(todos)


    def _todo_write_internal(self, todos: List[Dict[str, Any]]) -> str:

        #console_logger.log_text(f"_todo_write_internal: Arguments {todos} with type {type(todos)}")

        if (type(todos) == str):
            todos = json.loads(todos)
            #console_logger.log_text(f"after json.loads {todos} with type {type(todos)}")


        try:
            # Validate and create TodoItem objects
            todo_items:List[TodoItem] = []
            valid_statuses = {"pending", "in_progress", "completed"}
            valid_priorities = {"high", "medium", "low"}
            
            for i, todo_dict in enumerate(todos):
                # Validate required fields
                if not isinstance(todo_dict, dict):
                    return f"Error: Todo item {i+1} is not a dictionary"
                
                required_fields = ["id", "content", "status", "priority"]
                for field in required_fields:
                    if field not in todo_dict:
                        return f"Error: Todo item {i+1} missing required field '{field}'"
                
                # Validate status
                if todo_dict["status"] not in valid_statuses:
                    return f"Error: Todo item {i+1} has invalid status '{todo_dict['status']}'. Valid statuses: {', '.join(valid_statuses)}"
                
                # Validate priority
                if todo_dict["priority"] not in valid_priorities:
                    return f"Error: Todo item {i+1} has invalid priority '{todo_dict['priority']}'. Valid priorities: {', '.join(valid_priorities)}"
                
                # Create TodoItem
                todo_item = TodoItem(**todo_dict)
                todo_items.append(todo_item)
                self.todos = todo_items
        except Exception as e:
            return f"Error processing todos: {str(e)}"
        return f"Successfully processed {len(todo_items)} todos"

    def todo_summary_message(self) -> Message:
        """Create a summary message of the todos."""
        return Message(role="system", content=TODO_MODIFICATION_RESPONSE_MSG + self.todo_to_json())

    def __llm_completion_with_tools(self, messages: List[Message], tool_choice: Optional[str] = "auto",
                           max_iterations: int = 50 ) -> CompletionResult:
        """Complete a conversation with automatic tool execution."""

        self.llm_model = llm_router.get()

        # Debug live panel to print each round of messages
        # self.start_debugpanel()
        # self.post_debugpanel("New Query:", messages=messages)
        self.post_json("New Query", messages=messages, type="user")

        full_conversation = messages.copy()
        current_result: List[Message] = []
        has_tool_calls = False    
        data_logger = get_data_logger()

        user_facing_result = ""
        
        for iteration in range(max_iterations):
            # Get response from LLM
            full_conversation_with_todos = full_conversation.copy()
            if self.has_todos():
                full_conversation_with_todos.append(self.todo_summary_message())
                console_logger.log_text(f"Appending TODOs: {self.todo_to_json()}")
            last_response = self.llm_model.complete(messages=full_conversation_with_todos, tools=self.get_tool_schemas(),
                tool_choice=tool_choice)
            # print(f"Latency (sec) = {last_response.latency_seconds}")
            data_logger.log_stats(self.llm_model.get_model_name(), prompt_tokens=last_response.get_prompt_tokens(),
                 completion_tokens=last_response.get_completion_tokens(), latency_seconds=last_response.get_latency_seconds(),
                  operation="tool_call" if has_tool_calls else "completion")
            
            if last_response.content:
                user_facing_result = last_response.content
      
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
        
        # Display final result to console.
        self.post_json("LLM completion with tools result", current_result, type="from_llm")
        # self.stop_debugpanel()
        return CompletionResult(has_tool_calls=has_tool_calls, full_conversation=full_conversation,
                                 current_result=current_result, last_response=assistant_message,
                                 user_facing_result=user_facing_result)
   
    

    def todo_to_xml(self) -> str:
        """Convert TodoItem to XML format.
        
        Returns:
            str: XML representation of the TodoItem
        """
        if not self.todos:
            return "<todos></todos>"
            
        xml = ["<todos>"]
        for todo in self.todos:
            xml.append(f"  <todo id='{todo.id}'>")
            xml.append(f"    <content>{todo.content}</content>")
            xml.append(f"    <status>{todo.status}</status>") 
            xml.append(f"    <priority>{todo.priority}</priority>")
            xml.append("  </todo>")
        xml.append("</todos>")
        return "\n".join(xml)

    def todo_to_json(self) -> str:
        """Convert TodoItem to JSON format.
        
        Returns:
            str: JSON representation of the TodoItem
        """
        return "\n".join([t.model_dump_json() for t in self.todos])
    
    def _todo_write_response(self) -> str:
        log_str = "\n".join([t.model_dump_json() for t in self.todos])

        # Escape quotes and newlines for string concatenation
        todo_str = log_str.replace('"', '\\"').replace('\n', '\\n')
        return TODO_MODIFICATION_RESPONSE_MSG + "\n" + todo_str

        # TODO: Move to console_logger.py


# DONEs:
# 1. Add instance method to CwTask to add TODos and update TODOs. This will be registered as a tool with ToolCaller instance.
# 2. CwTask. register tools with ToolCaller instance. 
# TODOS:
# 1. Make LLM agentic call, log all requests. Check tools requests. Check Langfuse traces.
# 2. Include TODOs as system reminder message to LLM added at the end, check wilson-traces.
# 3. Run on large codebase and see differences
# 4. Allow llm agentic call as a tool, mimicing claude code.
# 5. Add a config with logs dir etc.


# TODOS; Llama Meta:
# 1. Create a set of coding evals, code compeletion, security reviews, code gen, bugfix gen, architecture.
# 2. Run evals.
#    - Create a set of eval runners for differnt datasets, and plotting tools.
#    - Compare against GPT-5, GPT-oss, Opus 4.1, Kimi 2, Llama 4 Scout, Maverick.

