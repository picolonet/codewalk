#!/usr/bin/env python3
"""Test script for KBBuilder"""

import sys
import os
sys.path.append('.')
sys.path.append('..')

# Add the cw directory to the path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cw'))

from cw.kb.kb_builder import KBBuilder
from cw.llm.lite_llm_model import LiteLlmModel

from dotenv import load_dotenv

load_dotenv("../.env")


def test_kb_builder():
    """Test the KBBuilder functionality"""
    
    # Create a test directory structure
    test_dir = "test_kb_project"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create some test files
    with open(f"{test_dir}/main.py", "w") as f:
        f.write("""
def main():
    '''Main function that starts the application'''
    print("Hello World")
    calculate_sum(1, 2)

def calculate_sum(a, b):
    '''Calculate sum of two numbers'''
    return a + b

if __name__ == "__main__":
    main()
""")
    
    with open(f"{test_dir}/utils.py", "w") as f:
        f.write("""
import os
import json

def read_config(file_path):
    '''Read configuration from JSON file'''
    with open(file_path, 'r') as f:
        return json.load(f)

def write_log(message):
    '''Write message to log file'''
    with open('app.log', 'a') as f:
        f.write(f"{message}\\n")
""")
    
    # Create .cw_ignore file
    with open(f"{test_dir}/.cw_ignore", "w") as f:
        f.write("*.log\n__pycache__\n.git\n")
    
    # Create subdirectory
    os.makedirs(f"{test_dir}/submodule", exist_ok=True)
    with open(f"{test_dir}/submodule/helper.py", "w") as f:
        f.write("""
class Helper:
    '''Helper class for utility functions'''
    
    def format_string(self, text):
        '''Format string with proper capitalization'''
        return text.title()
""")
    
    try:
        # Initialize LLM model (you may need to adjust this based on your setup)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        model_name = os.environ.get("ANTHROPIC_MODEL_NAME") or "gpt-3.5-turbo"

        model = LiteLlmModel(
            model=model_name,
            base_url=base_url,
            api_key=api_key,  # Or set OPENAI_API_KEY environment variable
            temperature=0.7)

        
        # Create KB builder
        kb_builder = KBBuilder(model)
        
        # Build knowledge base
        print("Building knowledge base...")
        kb_builder.build_knowledge_base(test_dir)
        
        print("Knowledge base build completed!")
        
        # Check if files were created
        kb_dir = f"{test_dir}/.cw_kb"
        if os.path.exists(kb_dir):
            print(f"KB directory created: {kb_dir}")
            
            # List generated files
            for root, dirs, files in os.walk(kb_dir):
                for file in files:
                    print(f"Generated: {os.path.join(root, file)}")
        else:
            print("ERROR: KB directory was not created")
            
    except Exception as e:
        print(f"Error during test: {e}")
    
    finally:
        # Clean up test directory
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print("Test directory cleaned up")

if __name__ == "__main__":
    test_kb_builder()