from typing import List, Dict, Any, Optional, Union
from cw.llm.llm_router import llm_router
from cw.util.data_logger import get_data_logger
from tool_caller import ToolCaller
from llm.llm_common import CompletionResult, LlmUtils, Message
from typing import Callable
from console_logger import console_logger

class LlmCompletion:
    def __init__(self, message_history: List[Message],
        tool_caller: ToolCaller, max_iterations: int = 50, aggregate_stats_key: str = "aggregate_key"):
        self.message_history = message_history
        self.max_iterations = max_iterations
        self.user_facing_result = ""
        self.tool_caller = tool_caller
        self.aggregate_stats_key = aggregate_stats_key

    def register_conversation_history_preprocessor(self, processor: Callable[[List[Message]], List[Message]]) -> None:
        """Register a callable that pre-processes and potentially modifies the conversation / message history before the LLM is called.
        
        Args:
            processor: A function that takes a List[Message] and returns a modified List[Message]
        """
        self.message_preprocessor = processor

    def complete(self, query: Message, tool_choice: Optional[str] = "auto", trace_name: Optional[str] = None, operation_tag: Optional[str] = None, **kwargs) -> CompletionResult:
        """Complete a conversation with automatic tool execution."""

        self.llm_model = llm_router().get()

        LlmUtils.post_json("New Query", messages=[query], type="user")

        full_conversation = self.message_history + [query]
        current_result: List[Message] = []
        has_tool_calls = False    
        data_logger = get_data_logger()

        self.user_facing_result = ""
        
        for iteration in range(self.max_iterations):
            # Get response from LLM
            if (self.message_preprocessor):
                full_conversation_preprocessed = self.message_preprocessor(full_conversation)
            else:
                full_conversation_preprocessed = full_conversation

            # full_conversation_with_todos = full_conversation.copy()
            # if self.has_todos():
            #     full_conversation_with_todos.append(self.todo_summary_message())
            #     console_logger.log_text(f"Appending TODOs: {self.todo_to_json()}")
            last_response = self.llm_model.complete(messages=full_conversation_preprocessed,
                tools=self.tool_caller.get_tool_schemas(),
                tool_choice=tool_choice, trace_name=trace_name)
            # print(f"Latency (sec) = {last_response.latency_seconds}")
            operation_tag = operation_tag or "completion"
            console_logger.log_text(f"Operation tag: {operation_tag}, trace_name: {trace_name}")
            data_logger.log_stats(self.llm_model.get_model_name(), prompt_tokens=last_response.get_prompt_tokens(),
                 completion_tokens=last_response.get_completion_tokens(), latency_seconds=last_response.get_latency_seconds(),
                  operation=operation_tag)
            data_logger.update_inmemory_stats(self.aggregate_stats_key, self.llm_model.get_model_name(),
                last_response.get_prompt_tokens(), last_response.get_completion_tokens(),
                last_response.get_latency_seconds(), operation=operation_tag)
            
            if last_response.content:
                user_facing_result = last_response.content
      
            # Add assistant message to conversation
            assistant_message = Message(
                role = "assistant",
                content = last_response.content,
                tool_calls = last_response.tool_calls
            )

            # Post to log message
            LlmUtils.post_json("Response from LLM:", [assistant_message], type="from_llm")

            full_conversation.append(assistant_message)
            current_result.append(assistant_message)
            
            
            # If no tool calls, we're done
            if not last_response.tool_calls:
                break
            has_tool_calls = True
            
            # Execute each tool call
            for tool_call in last_response.tool_calls:
                # Log tool calls.
                tool_response = self.tool_caller.execute_tool(tool_call)
                
                # Add tool response to conversation
                tool_message = Message(
                    role="tool",
                    content=tool_response.content,
                    tool_call_id=tool_response.tool_call_id
                )
                # log message: Tool call
                LlmUtils.post_json("Tool call output to LLM:", [tool_message], type = "to_llm")
                full_conversation.append(tool_message)
                current_result.append(tool_message)
        
        # Display final result to console.
        LlmUtils.post_json("LLM completion with tools result", current_result, type="from_llm")
        # self.stop_debugpanel()
        return CompletionResult(has_tool_calls=has_tool_calls, full_conversation=full_conversation,
                                 current_result=current_result, last_response=assistant_message,
                                 user_facing_result=user_facing_result)