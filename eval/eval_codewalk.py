#!/usr/bin/env python3
"""
Evaluation script for FastAPI Q&A using codewalk.
Loads eval_qa.yaml, runs codewalk queries, and appends results.
"""

import subprocess
import yaml
import os
import sys
from pathlib import Path


def load_qa_file(file_path):
    """Load the FastAPI Q&A YAML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_qa_file(data, file_path):
    """Save the updated FastAPI Q&A YAML file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def run_codewalk_query(question):
    """Run codewalk with the given question and return the output."""
    try:
        result = subprocess.run(
            ['codewalk', '--query', question],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode != 0:
            print(f"Error running codewalk: {result.stderr}")
            return None
            
        # Read the output from codewalk_out.txt
        output_file = Path('codewalk_out.txt')
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            print("Warning: codewalk_out.txt not found")
            return None
            
    except FileNotFoundError:
        print("Error: codewalk command not found. Make sure it's installed and in PATH.")
        return None
    except Exception as e:
        print(f"Error running codewalk: {e}")
        return None


def main():
    """Main evaluation function."""
    qa_file_path = Path(__file__).parent / 'eval_qa.yaml'
    
    if not qa_file_path.exists():
        print(f"Error: {qa_file_path} not found")
        sys.exit(1)
    
    print(f"Loading Q&A file: {qa_file_path}")
    qa_data = load_qa_file(qa_file_path)
    
    if 'questions' not in qa_data:
        print("Error: No 'questions' key found in YAML file")
        sys.exit(1)
    
    total_questions = len(qa_data['questions'])
    print(f"Found {total_questions} questions to process")
    
    for i, qa_item in enumerate(qa_data['questions'], 1):
        if 'question' not in qa_item:
            print(f"Skipping item {i}: no 'question' field")
            continue
            
        question = qa_item['question']
        print(f"Processing question {i}/{total_questions}: {question[:60]}...")
        
        # Skip if codewalk_answer already exists
        if 'codewalk_answer' in qa_item:
            print(f"  Skipping - codewalk_answer already exists")
            continue
        
        # Run codewalk query
        answer = run_codewalk_query(question)
        
        if answer:
            qa_item['codewalk_answer'] = answer
            print(f"  Added codewalk answer ({len(answer)} characters)")
        else:
            print(f"  Failed to get codewalk answer")
    
    # Save the updated file
    print(f"Saving updated Q&A file...")
    save_qa_file(qa_data, qa_file_path)
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
