#!/usr/bin/env python3
"""
Example of Langfuse v3 trace generation using the updated llm_model.py patterns.

This example shows how to:
1. Create a trace
2. Create a generation within the trace
3. End the generation with output and usage data
4. Optionally create multiple generations within the same trace
"""

import os
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv("../.env")


# Add the cw directory to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cw'))

from langfuse import Langfuse
from llm.llm_model import Message

# Initialize Langfuse client
langfuse = Langfuse(
    # These can be set via environment variables:
    # LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST
)

def example_simple_trace():
    """Example 1: Simple trace with one generation"""
    print("=== Example 1: Simple Trace ===")
    
    # 1. Create a trace
    trace = langfuse.trace(name="chat_completion_example")
    
    # 2. Create sample messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    # 3. Create a generation within the trace
    generation = trace.generation(
        name="openai_completion",
        model="gpt-3.5-turbo",
        input=messages,
        metadata={
            "temperature": 0.7,
            "max_tokens": 100,
            "tools": 0,
            "tool_choice": None
        }
    )
    
    # 4. Simulate LLM response
    response_content = "The capital of France is Paris."
    
    # 5. End the generation with output and usage
    generation.end(
        output=response_content,
        usage={
            "input": 25,      # prompt tokens
            "output": 8,      # completion tokens  
            "total": 33       # total tokens
        }
    )
    
    print(f"✅ Trace ID: {trace.id}")
    print(f"✅ Generation ID: {generation.id}")
    print(f"✅ Response: {response_content}")


def example_tool_calling_trace():
    """Example 2: Trace with tool calling"""
    print("\n=== Example 2: Tool Calling Trace ===")
    
    # 1. Create a trace for tool calling scenario
    trace = langfuse.trace(name="tool_calling_example")
    
    # 2. First generation - user query with tool available
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "What's the weather like in Paris?"}
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    
    generation1 = trace.generation(
        name="tool_call_request",
        model="gpt-4",
        input=messages,
        metadata={
            "temperature": 0.1,
            "max_tokens": 200,
            "tools": len(tools),
            "tool_choice": "auto"
        }
    )
    
    # Simulate tool call response
    tool_call_response = {
        "content": None,
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}'
                }
            }
        ]
    }
    
    generation1.end(
        output=tool_call_response,
        usage={"input": 85, "output": 25, "total": 110}
    )
    
    # 3. Second generation - tool response and final answer
    messages_with_tool = messages + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_call_response["tool_calls"]
        },
        {
            "role": "tool",
            "content": '{"temperature": 22, "condition": "sunny", "humidity": 60}',
            "tool_call_id": "call_123"
        }
    ]
    
    generation2 = trace.generation(
        name="final_response",
        model="gpt-4",
        input=messages_with_tool,
        metadata={"temperature": 0.1, "max_tokens": 200}
    )
    
    final_response = "The weather in Paris is currently sunny with a temperature of 22°C and 60% humidity."
    
    generation2.end(
        output=final_response,
        usage={"input": 120, "output": 18, "total": 138}
    )
    
    print(f"✅ Trace ID: {trace.id}")
    print(f"✅ Generation 1 ID: {generation1.id}")
    print(f"✅ Generation 2 ID: {generation2.id}")
    print(f"✅ Final Response: {final_response}")


def example_using_llm_model_methods():
    """Example 3: Using the methods from our LlmModel class"""
    print("\n=== Example 3: Using LlmModel Methods ===")
    
    # Create a mock LLM model instance to demonstrate the methods
    class MockLlmModel:
        def __init__(self):
            self.langfuse = Langfuse()
        
        def _create_trace(self, name: str):
            """Create a new trace in Langfuse v3."""
            return self.langfuse.trace(name=name)

        def _trace_generation(self, trace, name: str, model: str, formatted_messages: List[Dict[str, Any]],
            temperature: float, max_tokens: Optional[int] = None, tools: Optional[List[Dict[str, Any]]] = None, 
            tool_choice: Optional[str] = None):
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
    
    # Use the mock model
    model = MockLlmModel()
    
    # 1. Create trace
    trace = model._create_trace("llm_model_example")
    
    # 2. Prepare messages
    formatted_messages = [
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ]
    
    # 3. Create generation
    generation = model._trace_generation(
        trace=trace,
        name="explanation_request",
        model="gpt-4",
        formatted_messages=formatted_messages,
        temperature=0.7,
        max_tokens=300
    )
    
    # 4. Simulate response
    response = ("Quantum computing uses quantum mechanical phenomena like superposition "
               "and entanglement to process information in ways that classical computers cannot. "
               "Think of it as computing with quantum bits (qubits) that can be in multiple "
               "states simultaneously, allowing for parallel processing of possibilities.")
    
    # 5. End the generation
    model._trace_end(
        generation=generation,
        output=response,
        prompt_tokens=15,
        completion_tokens=45,
        total_tokens=60
    )
    
    print(f"✅ Trace ID: {trace.id}")
    print(f"✅ Generation ID: {generation.id}")
    print(f"✅ Response: {response[:100]}...")


def example_with_error_handling():
    """Example 4: Trace with error handling"""
    print("\n=== Example 4: Error Handling ===")
    
    trace = langfuse.trace(name="error_handling_example")
    
    try:
        generation = trace.generation(
            name="risky_operation",
            model="gpt-4",
            input=[{"role": "user", "content": "This might fail"}],
        )
        
        # Simulate an error occurring
        raise Exception("Simulated API error")
        
    except Exception as e:
        # Log the error in the generation
        generation.end(
            output=f"Error occurred: {str(e)}",
            level="ERROR",  # v3 supports logging levels
            metadata={"error_type": type(e).__name__}
        )
        
        print(f"✅ Error logged in trace: {trace.id}")
        print(f"✅ Generation ID: {generation.id}")
        print(f"✅ Error: {str(e)}")


if __name__ == "__main__":
    print("Langfuse v3 Trace Generation Examples")
    print("====================================")
    
    # Run all examples
    example_simple_trace()
    example_tool_calling_trace()
    example_using_llm_model_methods()
    example_with_error_handling()
    
    print("\n🎉 All examples completed! Check your Langfuse dashboard to see the traces.")
    print("💡 Make sure to set your Langfuse environment variables:")
    print("   - LANGFUSE_SECRET_KEY")
    print("   - LANGFUSE_PUBLIC_KEY")  
    print("   - LANGFUSE_HOST (optional, defaults to https://cloud.langfuse.com)")